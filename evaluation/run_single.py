import os
import json
import warnings
import argparse

from evaluation.models.model_zoo import get_model
from evaluation.eval_tools.asr import whisper_transcribe
from evaluation.eval_tools.emotion import emotion_to_vec_scores
from evaluation.eval_tools.quality import audio_quality_scores
from evaluation.eval_tools.properties import audio_properties

def main():
    parser = argparse.ArgumentParser(description="Run model inference and evaluation on an audio file.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file (.wav)")
    args = parser.parse_args()

    # Step 1: Load and run the model
    model = get_model(args.model)
    output_audio_path, generated_text = model(args.audio_path)

    # Step 2: Transcribe the audio
    transcription = whisper_transcribe(output_audio_path)

    # Step 3: Audio properties
    audio_prop = audio_properties(output_audio_path)

    # Step 4: Emotion scores
    emotion_scores = emotion_to_vec_scores(output_audio_path)

    # Step 5: Audio quality scores
    audio_scores = audio_quality_scores(output_audio_path, generated_text, transcription)

    # Step 6: Build output JSON
    output_data = {
        "agent_response": transcription,
        "agent_emotion": emotion_scores,
        "agent_audio_quality": audio_scores,
        "agent_audio_properties": audio_prop,
    }

    # Save JSON file
    json_output_path = args.audio_path.replace(".wav", f"_{args.model}_output.json")
    with open(json_output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nGenerated text:\n{generated_text}")
    print(f"Output audio saved at: {output_audio_path}")
    print(f"JSON saved to: {json_output_path}")

if __name__ == "__main__":
    main()
