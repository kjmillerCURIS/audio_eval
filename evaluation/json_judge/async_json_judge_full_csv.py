import os
import json
import csv
import asyncio
from string import Template
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from asyncio_throttle import Throttler  # to limit concurrency if needed

# === CONFIGURATION (fill these in) ===
ENTRIES_JSON_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/datasets/arjun_speakbench508_dataset.json"
AUDIO_JSON_BASE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments"
PROMPT_TEMPLATE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/json_judge/experiments_sample_200/pairwise_prompt_speakbench_multi_aspect.txt"
OUTPUT_CSV_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/json_judge/output_speakbench_gemini_multi_aspect.csv"

LLM_MODEL = "gemini-2.5-flash"  #  "gemini-2.5-flash" or "gpt-4o"
MAX_CONCURRENT_REQUESTS = 100



from dotenv import load_dotenv
load_dotenv("/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/.env")  


os.environ["OPENAI_API_KEY"] = os.getenv("GPT_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_KEY")
# =====================================

class Judgement(BaseModel):
    reasoning: str
    content: str
    voice_quality: str
    instruction_following_audio: str

# -----------------------------
# Helpers
# -----------------------------

def format_prompt(prompt_path, input_prompt, model_a_json, model_b_json):
    with open(prompt_path, "r") as f:
        template = Template(f.read())

    model_a_str = json.dumps(model_a_json, indent=2)
    model_b_str = json.dumps(model_b_json, indent=2)

    filled_prompt = template.substitute(
        user_prompt=input_prompt,
        model_a=model_a_str,
        model_b=model_b_str
    )
    return filled_prompt

def clean_agent_json(agent_json):
    agent_json.pop("agent_speaker_consistency", None)
    agent_json.get("agent_audio_quality", {}).pop("WER", None)
    agent_json.get("agent_audio_quality", {}).pop("UTMOSv2_Mean_Opinion_Score", None)
    agent_json.pop("agent_word_level_timestamps", None)
    return agent_json

async def evaluate_entry(entry: dict, model, throttler: Throttler) -> tuple:
    idx = entry.get("index", "?")
    model_a = entry.get("model_a", "")
    model_b = entry.get("model_b", "")
    
    # Default return in case of failure
    def empty_result(reason=""):
        return (
            idx, model_a, model_b, "", "", "", 
            entry.get("label", {}).get("content", ""),
            entry.get("label", {}).get("voice_quality", ""),
            entry.get("label", {}).get("instruction_following_audio", ""),
            reason
        )

    # Extract multi-dim labels
    content_label = entry.get("label", {}).get("content", "")
    vq_label = entry.get("label", {}).get("voice_quality", "")
    if_label = entry.get("label", {}).get("instruction_following_audio", "")

    # Convert audio paths to JSON paths
    audio1_json_path = os.path.join(AUDIO_JSON_BASE_PATH, entry.get("audio1_path", "").replace(".wav", ".json"))
    audio2_json_path = os.path.join(AUDIO_JSON_BASE_PATH, entry.get("audio2_path", "").replace(".wav", ".json"))

    # Check if files exist
    if not os.path.exists(audio1_json_path) or not os.path.exists(audio2_json_path):
        reason = f"Missing audio JSON: {audio1_json_path if not os.path.exists(audio1_json_path) else ''} {audio2_json_path if not os.path.exists(audio2_json_path) else ''}".strip()
        print(f"[WARN] {reason}")
        return empty_result(reason)

    # Load JSON safely
    try:
        with open(audio1_json_path, "r") as f1, open(audio2_json_path, "r") as f2:
            audio1_json = json.load(f1)
            audio2_json = json.load(f2)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON for index {idx}: {e}")
        return empty_result(f"JSON load error: {e}")

    # Clean
    audio1_json = clean_agent_json(audio1_json)
    audio2_json = clean_agent_json(audio2_json)

    prompt = format_prompt(PROMPT_TEMPLATE_PATH, entry.get("instruction_text", ""), audio1_json, audio2_json)

    # --- Async LLM call with throttler ---
    async with throttler:
        try:
            response = await model.ainvoke(prompt)
        except Exception as e:
            print(f"[ERROR] LLM call failed for index {idx}: {e}")
            return empty_result(f"LLM error: {e}")

    # Extract JSON
    response_text = response.content
    start, end = response_text.find('{'), response_text.rfind('}')
    if start >= 0 and end >= 0:
        try:
            llm_response = json.loads(response_text[start:end+1])
        except Exception as e:
            print(f"[ERROR] JSON parse failed for index {idx}: {e}")
            return empty_result(f"JSON parse error: {e}")
    else:
        llm_response = {}

    prediction_content = llm_response.get("content", "").strip()
    prediction_vq = llm_response.get("voice_quality", "").strip()
    prediction_if = llm_response.get("instruction_following_audio", "").strip()
    reasoning = llm_response.get("reasoning", "").strip()

    return (
        idx, model_a, model_b,
        prediction_content, prediction_vq, prediction_if,
        content_label, vq_label, if_label,
        reasoning
    )

# -----------------------------
# Main
# -----------------------------

async def main():
    # Load entries
    with open(ENTRIES_JSON_PATH, "r") as f:
        all_entries = json.load(f)

    # Init LLM
    model = init_chat_model(LLM_MODEL, model_provider="openai" if "gpt" in LLM_MODEL else "google_genai")

    # Throttler for concurrency
    throttler = Throttler(rate_limit=MAX_CONCURRENT_REQUESTS)

    # Run all evaluations in parallel
    tasks = [evaluate_entry(entry, model, throttler) for entry in all_entries]
    results = await asyncio.gather(*tasks)

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
    rows.extend(results)

    # Save CSV
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved results to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
