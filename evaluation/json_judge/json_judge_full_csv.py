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
ENTRIES_JSON_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/datasets/arjun_speakbench508_dataset.json"
AUDIO_JSON_BASE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments"
PROMPT_TEMPLATE_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/json_judge/experiments_sample_200/pairwise_prompt_speakbench_multi_aspect.txt"
OUTPUT_CSV_PATH = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/json_judge/output_speakbench_gpt_multi_aspect_2.csv"
# =====================================


class Judgement(BaseModel):
    reasoning: str
    content: str
    voice_quality: str
    instruction_following_audio: str

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
        # Load entries
    with open(ENTRIES_JSON_PATH, "r") as f:
        all_entries = json.load(f)

    judge = GPT4oJudge(GPT_KEY)

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

    # Clean helper
    def clean_agent_json(agent_json):
        agent_json.pop("agent_speaker_consistency", None)
        agent_json.get("agent_audio_quality", {}).pop("WER", None)
        agent_json.get("agent_audio_quality", {}).pop("UTMOSv2_Mean_Opinion_Score", None)
        agent_json.pop("agent_word_level_timestamps", None)
        return agent_json

    for entry in all_entries:
        try:
            idx = entry["index"]
            instruction_text = entry["instruction_text"]
            model_a = entry["model_a"]
            model_b = entry["model_b"]
            
            #Extract multi dim labels 
            content_label = entry["label"]["content"]
            vq_label = entry["label"]["voice_quality"]
            if_label = entry["label"]["instruction_following_audio"]


            # Convert audio paths to JSON paths
            audio1_json_path = os.path.join(AUDIO_JSON_BASE_PATH, entry["audio1_path"].replace(".wav", ".json"))
            audio2_json_path = os.path.join(AUDIO_JSON_BASE_PATH, entry["audio2_path"].replace(".wav", ".json"))

            # Load audio output JSONs
            with open(audio1_json_path, "r") as f1, open(audio2_json_path, "r") as f2:
                audio1_json = json.load(f1)
                audio2_json = json.load(f2)

            # Clean
            audio1_json = clean_agent_json(audio1_json)
            audio2_json = clean_agent_json(audio2_json)

            # --- First prompt (audio1 vs audio2) ---
            prompt1 = format_prompt(PROMPT_TEMPLATE_PATH, instruction_text, audio1_json, audio2_json)
            print(prompt1)
            llm_response1 = judge(prompt1, is_json=True)
            print(llm_response1)
            
            prediction_content = llm_response1.get("content", "").strip()
            prediction_vq = llm_response1.get("voice_quality", "").strip()
            prediction_if = llm_response1.get("instruction_following_audio", "").strip()

            reasoning = llm_response1.get("reasoning", "").strip()


            # --- Second prompt (audio2 vs audio1) ---
            # prompt2 = format_prompt(PROMPT_TEMPLATE_PATH, instruction_text, audio2_json, audio1_json)
            # print(prompt2)
            # llm_response2 = judge(prompt2, is_json=True)
            # print(llm_response2)
            # prediction_pos_2 = llm_response2.get("label", "").strip()

            # Ground truths
            #ground_truth_pos_1 = label
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
