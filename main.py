import os
import uuid

import cohere
import librosa
import soundfile as sf
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import pickle5 as pickle
from pydantic import BaseSettings, Field
import numpy as np


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


def tts_with_voice(text: str, voice_path: str):
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
    if not os.path.exists("output"):
        os.makedirs("output")

    synthesizer.save_wav(wav, f"output/{voice_path.split('/')[-1].split('.')[0]}.wav")


def generate_story(morale, main_character):
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
    generate_response = co.generate(
        model="command-xlarge-20221108",
        prompt=prompt,
        max_tokens=1400,
        temperature=1.9,
        k=450,
        p=0.9,
        frequency_penalty=0.25,
        presence_penalty=0.2,
        stop_sequences=[],
        return_likelihoods="NONE",
    )
    return generate_response.generations[0].text


def main():
    # Ask for file path
    audio_file = input("Enter file path: ")

    # Ask for a morale input
    input_morale = input("Enter a morale: ")
    input_main_character = input("Give the name of the main character: ")

    # save with uuid
    output_file = f"voices/{uuid.uuid4()}.wav"

    # Convert audio file
    convert_audio(audio_file, output_file)

    # Generate story
    story = generate_story(input_morale, input_main_character)

    # TTS with voice
    tts_with_voice(story, output_file)


if __name__ == "__main__":
    main()
