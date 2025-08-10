import json
import random
import os
from tqdm import tqdm
from pathlib import Path

# Import your audio analysis functions
from evaluation.models.model_zoo import get_model
from evaluation.eval_tools.asr import whisper_transcribe
from evaluation.eval_tools.emotion import emotion_to_vec_scores
from evaluation.eval_tools.quality import audio_quality_scores
from evaluation.eval_tools.properties import audio_properties
from evaluation.eval_tools.accent import get_accent
from evaluation.eval_tools.consistency import get_consistency_score

# Config
BASE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/"
NUM_SAMPLES = 200
SAMPLED_INDICES_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/gpt4o_test/sampled_indices_speakbench.json"  # <- Path to save the list of sampled indices

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
        "agent_emotion": emotion_scores,
        "agent_accent": accent_scores,
        "agent_audio_quality": audio_scores,
        "agent_audio_properties": audio_prop,
        "agent_speaker_consistency": cos_score,
    }

def process_json_list(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Randomly select entries
    sampled_entries = random.sample(data, NUM_SAMPLES)

    # Extract and save just the indices
    sampled_indices = [entry["index"] for entry in sampled_entries]
    with open(SAMPLED_INDICES_PATH, "w") as f:
        json.dump(sampled_indices, f, indent=2)
    print(f"Sampled indices saved to: {SAMPLED_INDICES_PATH}")

    # Flattened list of (entry, audio_key, transcript_key) tuples to feed into tqdm
    audio_jobs = [
        (entry, audio_key)
        for entry in sampled_entries
        for audio_key in ["audio1_path", "audio2_path"]
    ]

    # Process audio files with progress bar
    for entry, audio_key in tqdm(audio_jobs, desc="Processing audio files"):
        full_audio_path = os.path.join(BASE_PATH, entry[audio_key])
        output_json_path = full_audio_path.replace(".wav", ".json")

        try:
            output_data = process_audio(full_audio_path, "entry[transcript_key]")
            with open(output_json_path, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Saved JSON to: {output_json_path}")
        except Exception as e:
            print(f"Error processing {full_audio_path}: {e}")

# Example usage
if __name__ == "__main__":
    process_json_list("/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/datasets/speakbench508_dataset.json")
