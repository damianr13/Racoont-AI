import logging
import os
import uuid
from typing import Tuple

import cohere
import librosa
import soundfile as sf
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import pickle5 as pickle
from cohere.generation import Generations
from pydantic import BaseSettings, Field
import numpy as np

import streamlit as st
from readability import Readability

mnt_dir = os.environ.get("MNT_DIR", ".")


class Settings(BaseSettings):
    cohere_api_key: str = Field()

    class Config:
        env_file = ".env"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def convert_audio(
    audio_file: str,
    output_file: str,
    sample_rate: int = 22050,
    overwrite: bool = False,
) -> str:
    if not overwrite and os.path.exists(output_file):
        return output_file

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Load audio file
    audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)

    # Convert to wav format
    sf.write(output_file, audio, sample_rate, subtype="PCM_16")

    return output_file


def tts_with_voice(text: str, voice_path: str) -> str:
    tts_manager = ModelManager()
    model_name = "tts_models/multilingual/multi-dataset/your_tts"
    model_path, config_path, model_item = tts_manager.download_model(model_name)

    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        tts_speakers_file=voice_path,
        use_cuda=False,
    )

    wav = synthesizer.tts(text=text, speaker_wav=voice_path, language_name="en")
    if not os.path.exists(f"{mnt_dir}/output"):
        os.makedirs(f"{mnt_dir}/output")

    story_file = f"{mnt_dir}/output/{voice_path.split('/')[-1].split('.')[0]}.wav"
    synthesizer.save_wav(wav, story_file)
    return story_file


def launch_generation(morale, main_character):
    with open("resources/fables-with-emb.pkl", "rb") as f:
        resource = pickle.load(f)

    fables = resource["fables"]
    fables_emb = resource["embs"]

    settings = Settings()
    co = cohere.Client(settings.cohere_api_key)
    morale_embed = co.embed([morale]).embeddings[0]

    distances = [
        cosine_similarity(np.array(e), np.array(morale_embed)) for e in fables_emb
    ]
    closest_fable = fables[np.argmax(distances)]

    prompt = f"""
        This is a short fable. 
        "{closest_fable}" 
        
        Based on it, create a longer story that expands over multiple paragraphs. Each character should have a name \
        and the final story should include more characters, but keep the same plot line. Use simple language, as this \
        should be a children bed time story.
        Name of the main character: {main_character}
        
        Once upon a time
    """
    return co.generate(
        model="command-xlarge-20221108",
        prompt=prompt,
        max_tokens=900,
        temperature=1.4,
        k=450,
        p=0.9,
        frequency_penalty=0.25,
        presence_penalty=0.2,
        stop_sequences=[],
        num_generations=5,
        return_likelihoods="NONE",
    )


def generate_story(generate_response: Generations):
    texts = [
        g.text
        for g in generate_response.generations
        if len(g.text.split(" ")) >= 100 and g.text.endswith(".")
    ]
    if not texts:
        # whatever, return the first one
        print("Warn: no suitable generation found")
        return generate_response.generations[0].text

    readability_scores = [Readability(t).spache().score for t in texts]
    return texts[np.argmax(readability_scores)]


def perform_generation(audio_file, story) -> str:
    st.session_state.disabled = True

    # save with uuid
    output_file = f"{mnt_dir}/voices/{uuid.uuid4()}.wav"

    # Convert audio file
    convert_audio(audio_file, output_file)

    # TTS with voice
    return tts_with_voice(story, output_file)


def main():
    if "story_details_submitted" not in st.session_state:
        st.session_state.story_details_submitted = False

    if "voice_submitted" not in st.session_state:
        st.session_state.voice_submitted = False

    # Ask for file path
    with st.form("user_input"):
        st.write("Please enter the following information")

        # Ask for a morale input
        st.session_state.input_morale = st.text_input(
            "What morale would you like to convey?"
        )
        st.session_state.input_main_character = st.text_input(
            "Give the name of the main character: "
        )

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state.story_details_submitted = True

    if st.session_state.story_details_submitted:
        generate_response = launch_generation(
            st.session_state.input_morale, st.session_state.input_main_character
        )
        with st.form("voice_input"):
            st.write(
                "Provide a voice sample by reading out a story. At least 1 min."
                "\n\n"
                "Tip: You can read out a page from a book"
                "\n\n"
                "Or select the default voice."
            )
            st.session_state.audio_file = st.file_uploader(
                "Upload a recording of your voice", type="wav"
            )
            st.session_state.default_voice = st.checkbox(
                "Use the default voice", value=False
            )
            submitted = st.form_submit_button("Submit")

            if submitted:
                if st.session_state.default_voice:
                    st.session_state.audio_file = "resources/voice.wav"
                st.session_state.voice_submitted = True

        if st.session_state.voice_submitted:
            with st.spinner("Generating story..."):
                st.session_state.disabled = True
                story = "Once upon a time, " + generate_story(generate_response)
                st.write(
                    story,
                )

                story_file = perform_generation(st.session_state.audio_file, story)
                st.session_state.disabled = False

                st.audio(story_file)


if __name__ == "__main__":
    main()
