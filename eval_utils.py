import os
import sys
import json
from tqdm import tqdm
print('transformer imports...')
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
print('yay transformers!')
from llm_utils import run_llm, fill_out_prompt


''' Includes both the eval-tools and the evaluator '''


def setup_transcription_pipe(model_id='openai/whisper-large-v3'):
    print('load transcription model...')
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    print('load transcription processor...')
    processor = AutoProcessor.from_pretrained(model_id)

    print('make transcription pipe...')
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype="auto",
        device_map="auto",
    )

    return pipe


def setup_eval_tool_models():
    models = {}
    models['transcription_pipe'] = setup_transcription_pipe()
    return models


def whisper_transcribe(audio_path, eval_tool_models):
    """
    Transcribes audio and returns:
    - transcript: full text
    - word_timestamps: list of {text, timestamp=(start, end)}
    """

    transcription = eval_tool_models['transcription_pipe'](
        audio_path,
        return_timestamps="word",
        generate_kwargs={"language": "english"}
    )

    transcript = transcription["text"]
    word_chunks = transcription["chunks"]

    transcript = transcript.strip()
    word_chunks[0]['text'] = word_chunks[0]['text'].lstrip()
    word_chunks[-1]['text'] = word_chunks[-1]['text'].rstrip()

    return {'transcript' : transcript, 'word_chunks' : word_chunks}


#returns JSON
#just transcriber for now...
def run_eval_tools(audio_path, eval_tool_models):
    eval_tool_outputs = {}
    transcriber_outputs = whisper_transcribe(audio_path, eval_tool_models)
    eval_tool_outputs['transcriber_outputs'] = transcriber_outputs
    return eval_tool_outputs


def run_evaluator_on_text(transcriber_outputs, text_eval_instructions, user_utterance, knowledge_base, setting_name, user_name):
    text_eval_outputs = {}
    with open('prompts/evaluator_text_hard_skills_prompt.txt', 'r') as f:
        hard_skills_prompt_template = f.read()

    with open('prompts/evaluator_text_soft_skills_prompt.txt', 'r') as f:
        soft_skills_prompt_template = f.read()

    #hard skills
    text_eval_outputs['hard_skills'] = []
    for goal in tqdm(text_eval_instructions['hard_skill_eval_instructions']):
        hard_skills_prompt = fill_out_prompt(hard_skills_prompt_template, setting_name=setting_name, user_name=user_name, user_query_json_str=user_utterance, voice_assistant_text=transcriber_outputs['transcript'], hard_skill_goal=goal, knowledge_base=knowledge_base)
        output, _ = run_llm(hard_skills_prompt, is_json=True)
        output['goal'] = goal
        text_eval_outputs['hard_skills'].append(output)

    #soft skills
    text_eval_outputs['soft_skills'] = []
    for behavior in tqdm(text_eval_instructions['soft_skill_eval_instructions']):
        soft_skills_prompt = fill_out_prompt(soft_skills_prompt_template, setting_name=setting_name, user_name=user_name, user_query_json_str=user_utterance, voice_assistant_text=transcriber_outputs['transcript'], soft_skill_expected_behavior=behavior)
        output, _ = run_llm(soft_skills_prompt, is_json=True)
        output['behavior'] = behavior
        text_eval_outputs['soft_skills'].append(output)

    return text_eval_outputs


#returns JSON
#just text for now...
#user_utterance and knowledge_base should be strings (of JSON objects)
def run_evaluator(eval_tool_outputs, eval_instructions, user_utterance, knowledge_base, setting_name, user_name):
    evaluator_outputs = {}
    transcriber_outputs, text_eval_instructions = eval_tool_outputs['transcriber_outputs'], eval_instructions['text_eval_instructions']
    evaluator_outputs['text_eval_outputs'] = run_evaluator_on_text(transcriber_outputs, text_eval_instructions, user_utterance, knowledge_base, setting_name, user_name)
    return evaluator_outputs


if __name__ == '__main__':
    audio_path = 'outputs_OLD/hospital_0_A_qwen_response.wav'
    eval_tool_models = setup_eval_tool_models()
    print('run whisper...')
    whisper_transcribe(audio_path, eval_tool_models)
