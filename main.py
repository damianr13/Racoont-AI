import os
import uuid

import librosa
import soundfile as sf
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


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


def main():
    # Ask for file path
    audio_file = input("Enter file path: ")

    # save with uuid
    output_file = f"voices/{uuid.uuid4()}.wav"

    # Convert audio file
    convert_audio(audio_file, output_file)

    # TTS with voice
    tts_with_voice("Hello world", output_file)


if __name__ == "__main__":
    main()
