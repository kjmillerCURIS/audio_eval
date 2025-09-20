from audiojudge import AudioJudge
import json 
import os
from dotenv import load_dotenv

load_dotenv("/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/.env")  

GEMINI_KEY = os.getenv("GEMINI_KEY")
GPT_KEY = os.getenv("GPT_KEY")


SYSTEM_PROMPT_NO_ICL = """
You are an evaluator of audio outputs produced by different audio-capable large language models. Your task is to compare two audio responses (Audio 1 and Audio 2) generated according to a user's instruction. 
Evaluate based on these criteria: 
1. Semantics: Does the content fulfill the user's request accurately? 
2. Paralinguistics: How well does the speech match requested tone, emotion, style, pacing, and expressiveness? 
Important: Do not favor verbalized descriptions of tone over actual tonal expression. A response that says "I am speaking excitedly" but sounds flat should rank lower than one that genuinely sounds excited. 
Follow this process: 
1. Analyze the key characteristics requested in the user's instruction 
2. Evaluate how well Audio 1 performs on these characteristics 
3. Evaluate how well Audio 2 performs on these characteristics 
4. Compare their strengths and weaknesses 
5. Decide which is better overall 
Avoid position bias and don't let response length influence your evaluation. After your analysis, output valid JSON with exactly two keys: 
'reasoning' (your explanation of the comparison) and 'label' (a string value: '1' if the first audio is better, '2' if the second audio is better, or 'tie' if they are equally good/bad. Please use \"tie\" sparingly, and only when you absoultely cannot choose the winner.)
"""

USER_PROMPT = """
Please analyze which of the two recordings follows the instruction better, or tie. Respond ONLY in text and output valid JSON with keys
'reasoning' and 'label' (string, '1', '2' or 'tie').
"""

def extract_json(response):
        start, end = response.find('{'), response.rfind('}')
        assert start >= 0 and end >= 0, f"Could not find JSON in response: {response}"
        json_str = response[start:end + 1]
        return json.loads(json_str)

# Initialize with API keys
judge = AudioJudge(
    openai_api_key=GPT_KEY,
    google_api_key=GEMINI_KEY
)

# Simple pairwise comparison
result = judge.judge_audio(
    audio1_path="/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/audio_data/speakbench508_audio/0/audio_a.wav",
    audio2_path="/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/audio_data/speakbench508_audio/0/audio_b.wav",
    instruction_path="/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/audio_data/speakbench508_audio/0/instruction.wav",
    system_prompt=SYSTEM_PROMPT_NO_ICL,
    user_prompt=USER_PROMPT,
    concatenation_method="no_concatenation",
    model="gemini-2.5-flash"
)

print(extract_json(result["response"]))