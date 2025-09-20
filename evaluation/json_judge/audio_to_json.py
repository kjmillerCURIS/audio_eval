import json
import random
import os
from tqdm import tqdm
from pathlib import Path

# Import audio analysis tools
from evaluation.eval_tools.asr import whisper_transcribe
from evaluation.eval_tools.emotion import emotion_to_vec_scores
from evaluation.eval_tools.quality import audio_quality_scores
from evaluation.eval_tools.properties import audio_properties
from evaluation.eval_tools.accent import get_accent
from evaluation.eval_tools.consistency import get_consistency_score

# Config
BASE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/"

def process_audio(audio_path, generated_text):
    """
    Run the full analysis pipeline on a single audio file.
    Returns a dictionary to save as JSON.
    """
    transcription, word_chunks = whisper_transcribe(audio_path)
    audio_prop = audio_properties(audio_path, word_chunks)
    emotion_scores = emotion_to_vec_scores(audio_path)
    accent_scores = get_accent(audio_path)
    audio_scores = audio_quality_scores(audio_path, generated_text, transcription)
    _, cos_score = get_consistency_score(audio_path)

    return {
        "agent_response": transcription,
        "agent_word_level_timestamps": word_chunks,
        "agent_emotion": emotion_scores,
        "agent_accent": accent_scores,
        "agent_audio_quality": audio_scores,
        "agent_audio_properties": audio_prop,
        "agent_speaker_consistency": cos_score,
    }


def process_json_list(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Flattened list of (entry, audio_key) tuples for tqdm
    audio_jobs = [
        (entry, audio_key)
        for entry in data
        for audio_key in ["audio1_path", "audio2_path"]
    ]

    # Process audio files with progress bar
    for entry, audio_key in tqdm(audio_jobs, desc="Processing audio files"):
        full_audio_path = os.path.join(BASE_PATH, entry[audio_key])
        output_json_path = full_audio_path.replace(".wav", ".json")

        try:
            output_data = process_audio(full_audio_path, generated_text="Dummy text")
            with open(output_json_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Saved JSON to: {output_json_path}")
        except Exception as e:
            print(f"Error processing {full_audio_path}: {e}")


# Example usage
if __name__ == "__main__":
    process_json_list("/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/datasets/speakbench508_dataset.json")
