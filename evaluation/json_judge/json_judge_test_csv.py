import os
import sys
import json
import csv
from string import Template

from openai import OpenAI
from google import genai
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv("/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/.env")  

GEMINI_KEY = os.getenv("GEMINI_KEY")
GPT_KEY = os.getenv("GPT_KEY")


# === CONFIGURATION (fill these in) ===
ENTRIES_JSON_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/datasets/speakbench508_dataset.json"
AUDIO_JSON_BASE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments"
INDICES_JSON_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/json_judge/experiments/sampled_indices_speakbench_200.json"
PROMPT_TEMPLATE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/json_judge/experiments/pairwise_prompt_speakbench_6.txt"
OUTPUT_CSV_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/json_judge/output_speakbench_gemini_6.csv"
# =====================================

class Judgement(BaseModel):
    reasoning: str
    label: str

class Gemini25FlashJudge:

    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"  #gemini-2.5-flash-preview-04-17 is not available

    def _extract_json(self, response_text):
        start, end = response_text.find('{'), response_text.rfind('}')
        assert start >= 0 and end >= 0, f"Could not find JSON in response: {response_text}"
        json_str = response_text[start:end + 1]
        return json.loads(json_str)

    def _run_helper(self, query):
        response = self.client.models.generate_content(
            model=self.model_name, 
            contents=query,
            config={
                "response_mime_type": "application/json",
                "response_schema": Judgement,
            },
        )
        return response.text

    def __call__(self, query, is_json=False):
        response = self._run_helper(query)
        if is_json:
            return self._extract_json(response)
        else:
            return response


class GPT4oJudge:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o"

    def _extract_json(self, response):
        start, end = response.find('{'), response.rfind('}')
        assert start >= 0 and end >= 0, f"Could not find JSON in response: {response}"
        json_str = response[start:end + 1]
        return json.loads(json_str)

    def _run_helper(self, query):
        messages = [{"role": "system", "content": query}]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def __call__(self, query, is_json=False):
        response = self._run_helper(query)
        if is_json:
            return self._extract_json(response)
        else:
            return response
    


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


def main():
    # Load entries and selected indices
    with open(ENTRIES_JSON_PATH, "r") as f:
        all_entries = json.load(f)

    with open(INDICES_JSON_PATH, "r") as f:
        selected_indices = set(json.load(f))

    #judge = GPT4oJudge(GPT_KEY)
    judge = Gemini25FlashJudge(GEMINI_KEY)

    # Map index -> entry for fast lookup
    entry_lookup = {entry["index"]: entry for entry in all_entries}

    # Prepare CSV output
    rows = [("index", "model_a", "model_b", "prediction", "ground_truth")]

    for idx in selected_indices:
        entry = entry_lookup.get(idx)
        if entry is None:
            print(f"Index {idx} not found in entries.")
            continue

        try:
            instruction_text = entry["instruction_text"]
            model_a = entry["model_a"]
            model_b = entry["model_b"]
            label = entry["label"]

            # Convert audio paths to JSON paths
            audio1_json_path = os.path.join(AUDIO_JSON_BASE_PATH, entry["audio1_path"].replace(".wav", ".json"))
            audio2_json_path = os.path.join(AUDIO_JSON_BASE_PATH, entry["audio2_path"].replace(".wav", ".json"))

            # Load audio output JSONs
            with open(audio1_json_path, "r") as f1, open(audio2_json_path, "r") as f2:
                audio1_json = json.load(f1)
                audio2_json = json.load(f2)

            # Remove any keys 
            def clean_agent_json(agent_json):
                agent_json.pop("agent_speaker_consistency", None)
                agent_json["agent_audio_quality"].pop("WER", None)  
                agent_json.pop("agent_word_level_timestamps", None)
                #agent_json.pop("agent_accent", None)  # Remove agent_accent if present
                # agent_json["agent_audio_properties"].pop("Full_Pitch_Contour_Hz", None)
                # agent_json["agent_audio_properties"].pop("Full_Loudness_Contour_LUFS", None)


                return agent_json

            # Clean both agent JSONs
            audio1_json = clean_agent_json(audio1_json)
            audio2_json = clean_agent_json(audio2_json)

            # Format prompt and query LLM
            prompt = format_prompt(PROMPT_TEMPLATE_PATH, instruction_text, audio1_json, audio2_json)
            print(prompt)
            llm_response = judge(prompt, is_json=True)

            print(llm_response)

            prediction = llm_response.get("label", "").strip()

            rows.append((idx, model_a, model_b, prediction, label))
            print(f"Processed index {idx}: predicted={prediction}, true={label}")
        except Exception as e: 
            print(f"Error processing index {idx}: {e}")



    # Save results
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved results to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
