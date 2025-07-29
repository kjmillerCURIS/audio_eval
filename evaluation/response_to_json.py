import os
import json
import argparse
from evaluation.eval_tools.asr import whisper_transcribe
from evaluation.eval_tools.emotion import emotion_to_vec_scores
from evaluation.eval_tools.quality import audio_quality_scores
from evaluation.eval_tools.properties import audio_properties


def test():
    parser = argparse.ArgumentParser(description="Transcribe an audio file and save to JSON.")
    parser.add_argument("--audio_path", required=True, help="Path to input audio file (.wav)")
    args = parser.parse_args()

    mos_score = dnsmos_score(args.audio_path)

    print(mos_score)


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file and save to JSON.")
    parser.add_argument("--audio_path", required=True, help="Path to input audio file (.wav)")
    args = parser.parse_args()

    #1.Transcribe voice assistant response
    transcription = whisper_transcribe(args.audio_path)

    #2. Basic Audio Properties
    audio_prop = audio_properties(args.audio_path)

    #3. Emotion classifier 
    emotion_scores = emotion_to_vec_scores(args.audio_path)
    
    #4. Sound Quality 
    audio_scores = audio_quality_scores(args.audio_path)

    

    # Save JSON
    output_path = args.audio_path.replace(".wav", ".json")

    output_data = {
        "agent_response": transcription,
        "agent_emotion": emotion_scores,
        "agent_audio_quality": audio_scores,
        "agent_audio_properties": audio_prop,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"JSON saved to: {output_path}")

if __name__ == "__main__":
    main()
    #test()