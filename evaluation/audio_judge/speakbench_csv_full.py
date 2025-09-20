import os
import sys
import json
import csv
from string import Template

from openai import OpenAI
from google import genai
from pydantic import BaseModel
from audiojudge import AudioJudge

from dotenv import load_dotenv
load_dotenv("/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/.env")  

GEMINI_KEY = os.getenv("GEMINI_KEY")
GPT_KEY = os.getenv("GPT_KEY")

SYSTEM_PROMPT_NO_ICL = """
You are an evaluator of audio outputs produced by different audio-capable large language models. Your task is to compare two audio
responses (Audio 1 and Audio 2) generated according to a user’s instruction. Evaluate based on these criteria: 
1. Content
- Does the content fulfill the user’s request accurately? 
- Did the content of the response appropriately address the user's instruction? 
2. Voice Quality 
- How good is the voice quality of the response?
- Does it sound natural/human, does it mispronounce words, does it have pops or echoes?
3. Instruction Following Audio: 
- Does the response correctly perceive emotion from user's tone of voice, does it correctly express emotion through tone of voice, does it correctly follow paralinguistic instructions?
- This includes both implicit audio instruction like emotional intelligence and explicit audio instruction following 

Avoid position bias and don’t let response length influence your evaluation. After your analysis, output valid JSON with exactly 4 keys:
- "reasoning": your explanation of the comparison along each dimension
- "content": your rating for content dimension. a string value ’1’ if the first audio is better, ’2’ if the second audio is better, 'both_bad' if they are equally bad, or 'both_good' if they are equally good
- "voice_quality": your rating for voice quality dimension. a string value ’1’ if the first audio is better, ’2’ if the second audio is better, 'both_bad' if they are equally bad, or 'both_good' if they are equally good
- "instruction_following_audio": your rating for instruction following audio dimension. a string value ’1’ if the first audio is better, ’2’ if the second audio is better, 'both_bad' if they are equally bad, or 'both_good' if they are equally good

You should only pick a winner along each dimension if they is a clear and obvious difference between the quality of the two responses. If it comes down to minor details, 
then you should opt for using 'both_bad' or 'both_good' instead.
"""

USER_PROMPT = """
Respond ONLY in text and output valid JSON with keys "reasoning", "content", "voice_quality", and "instruction_following_audio":
"""


# === CONFIGURATION (fill these in) ===
ENTRIES_JSON_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/datasets/arjun_speakbench508_dataset.json"
AUDIO_JSON_BASE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments"
OUTPUT_CSV_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/audio_judge/experiments_full/output_speakbench_gemini_multi_aspect.csv"
# =====================================


def extract_json(response):
        start, end = response.find('{'), response.rfind('}')
        assert start >= 0 and end >= 0, f"Could not find JSON in response: {response}"
        json_str = response[start:end + 1]
        return json.loads(json_str)

def main():
        # Load entries
    with open(ENTRIES_JSON_PATH, "r") as f:
        all_entries = json.load(f)

    # Initialize with API keys
    judge = AudioJudge(
        openai_api_key=GPT_KEY,
        google_api_key=GEMINI_KEY
    )

    # Prepare CSV output
    rows = [(
        "index",
        "model_a",
        "model_b",
        "prediction_content", 
        "prediction_vq",
        "prediction_if",
        "gt_content", 
        "gt_vq",
        "gt_if",
        "reasoning"
    )]

    def swap_label(label: str) -> str:
        """Swap labels for position 2 ground truth."""
        if label == "1":
            return "2"
        elif label == "2":
            return "1"
        else:  # "tie"
            return "tie"


    for entry in all_entries:
        try:
            idx = entry["index"]
            model_a = entry["model_a"]
            model_b = entry["model_b"]
            
            #Extract multi dim labels 
            content_label = entry["label"]["content"]
            vq_label = entry["label"]["voice_quality"]
            if_label = entry["label"]["instruction_following_audio"]

            # Instruction and audio paths
            audio1_path = os.path.join(AUDIO_JSON_BASE_PATH, entry["audio1_path"])
            audio2_path = os.path.join(AUDIO_JSON_BASE_PATH, entry["audio2_path"])
            instruction_path = os.path.join(AUDIO_JSON_BASE_PATH, entry["instruction_path"])

            # --- First prompt (audio1 vs audio2) ---
            result_1 = judge.judge_audio(
                audio1_path=audio1_path,
                audio2_path=audio2_path,
                instruction_path=instruction_path,
                system_prompt=SYSTEM_PROMPT_NO_ICL,
                user_prompt=USER_PROMPT,
                concatenation_method="no_concatenation",
                model="gemini-2.5-flash"
            )

            response_1 = extract_json(result_1["response"])
            print(response_1)

            prediction_content = response_1.get("content", "").strip()
            prediction_vq = response_1.get("voice_quality", "").strip()
            prediction_if = response_1.get("instruction_following_audio", "").strip()

            reasoning = response_1.get("reasoning", "").strip()

       

            # --- Second prompt (audio2 vs audio1) ---
            # result_2 = judge.judge_audio(
            #     audio1_path=audio2_path,
            #     audio2_path=audio1_path,
            #     instruction_path=instruction_path,
            #     system_prompt=SYSTEM_PROMPT_NO_ICL,
            #     user_prompt=USER_PROMPT,
            #     concatenation_method="no_concatenation",
            #     model="gemini-2.5-flash"
            # )

            # response_2 = extract_json(result_2["response"])
            # print(response_2)
            # prediction_pos_2 = response_2.get("label", "").strip()

            # # Ground truths
            # ground_truth_pos_1 = label
            # ground_truth_pos_2 = swap_label(label)

            rows.append((
                idx,
                model_a,
                model_b,
                prediction_content,
                prediction_vq,
                prediction_if,
                content_label,
                vq_label,
                if_label,
                reasoning
            ))

            print(f"Processed index {idx}")

        except Exception as e:
            print(f"Error processing index {entry.get('index', '?')}: {e}")

    # Save results
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved results to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
