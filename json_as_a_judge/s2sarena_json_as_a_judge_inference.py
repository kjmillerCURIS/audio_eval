import os
import sys
import copy
import json
import numpy as np
from string import Template
from tqdm import tqdm
sys.path.append('.')
from json_as_a_judge.s2sarena_utils import load_example_info_dict
from json_as_a_judge.models.model_zoo import get_model
from json_as_a_judge.eval_tools.asr import whisper_transcribe
from json_as_a_judge.eval_tools.asr_v2 import whisper_transcribe as whisper_transcribe_v2
from json_as_a_judge.eval_tools.emotion import emotion_to_vec_scores
from json_as_a_judge.eval_tools.accent import get_accent
from json_as_a_judge.eval_tools.quality import audio_quality_scores
from json_as_a_judge.eval_tools.properties import audio_properties
from json_as_a_judge.eval_tools.properties_v2 import audio_properties as audio_properties_v2
from json_as_a_judge.eval_tools.consistency import get_consistency_score
from llm_utils import run_llm, SUPPORTED_LLM_NAMES


USE_SAVED_PROGRESS = True
#BAD_KEYS = ['33', '47', '55', '58', '76', '102', '130', '308']
BAD_KEYS = []
FULL_JSON_DESCRIPTION = \
{
  "user/agent_text_transcription": "a text transcription of the speech, obtained using Whisper",
  "user/agent_emotion": "a vector of emotion scores for the speech from the emotion2vec model",
  "user/agent_audio_quality": {
    "UTMOSv2_Mean_Opinion_Score": "Mean opinion score from UTMOSv2 model (1-5, higher is better)",
    "DNSMOS_Personalized_Signal_Quality": "Signal quality score from DNSMOS model (1-5, higher is better)",
    "DNSMOS_Personalized_Background_Quality": "Background noise quality score from DNSMOS model (1-5, higher is better)",
    "DNSMOS_Personalized_Overall_Quality": "Overall naturalness and audio quality score from DNSMOS model (1-5, higher is better)",
    "P808_Overall_Quality": "Overall naturalness and audio quality score from P.808 recommendation standard (1-5, higher is better)",
  },
  "user/agent_aggregate_audio_quality": "Average of audio quality scores from several off-the-shelf models (1-5, higher is better)",
  "user/agent_audio_properties": {
    "Mean_Pitch_Hz": "Mean pitch (fundamental frequency) of speech",
    "Std_Dev_Pitch_Hz": "Standard deviation in pitch",
    "Mean_RMS_dBFS": "Mean root mean squared decibels full scale",
    "Speech_Rate_WPM": "speech rate in words per minute",
    "Articulation_Rate_WPM": "speech rate in words per minute excluding pauses and gaps in speech",
  },
  "user/agent_speaker_consistency": "consistency in agent speaker identity, measured using cosine similarity between speaker embeddings in adjacent audio chunks"
}

FULL_JSON_DESCRIPTION_V2 = \
{
  "user/agent_text_transcription": "a text transcription of the speech, obtained using Whisper",
  "user/agent_emotion": "a vector of emotion scores for the speech from the emotion2vec model",
  "user/agent_accent": "a vector of cosine similarity scores for the agent's accent",
  "user/agent_audio_quality": {
    "DNSMOS_Personalized_Signal_Quality": "Signal quality score from DNSMOS model (1-5, higher is better)",
    "DNSMOS_Personalized_Background_Quality": "Background noise quality score from DNSMOS model (1-5, higher is better)",
    "DNSMOS_Personalized_Overall_Quality": "Overall naturalness and audio quality score from DNSMOS model (1-5, higher is better)",
    "P808_Overall_Quality": "Overall naturalness and audio quality score from P.808 recommendation standard (1-5, higher is better)",
  },
  "user/agent_audio_properties": {
    "Mean_Pitch_Hz": "Mean pitch (fundamental frequency) of speech",
    "Std_Dev_Pitch_Hz": "Standard deviation in pitch",
    "Full_Pitch_Contour_Hz": "full pitch contour",
    "Integrated_Loudness_LUFS": "average loudness of the agent response measured in LUFS",
    "Std_Dev_Loudness_LUFS": "standard deviation in loudness",
    "Full_Loudness_Contour_LUFS": "full loudness contour",
    "Speech_Rate_WPM": "speech rate in words per minute",
    "Articulation_Rate_WPM": "speech rate in words per minute excluding pauses and gaps in speech",
  }
}

GUIDANCE_HEADER = 'Some importance guidance for evaluating the responses:'
SEMANTIC_GUIDANCE = 'Correctness and relevance are the top priority, unless the input explicitly requests particular paralinguistics. It is better to say "I don\'t know" than to give an incorrect or irrelevant answer. It is not good for an agent to guess or assume facts that aren\'t known from the user\'s input. It is better to give broad or conditional adivce than to give advice that\'s based on guesses or assumptions.'
PROPERTIES_GUIDANCE = 'Do not make assumptions about the user\'s preferences or perceptions regarding speech rate or pitch, unless explicitly stated, as these can vary from person to person.'
GUIDANCE_DICT = {
        'none': '',
        'semantics': f'{GUIDANCE_HEADER}\n * {SEMANTIC_GUIDANCE}\n',
        'semantics_and_properties': f'{GUIDANCE_HEADER}\n * {SEMANTIC_GUIDANCE}\n * {PROPERTIES_GUIDANCE}\n',
        'properties': f'{GUIDANCE_HEADER}\n * {PROPERTIES_GUIDANCE}\n'
        }

PARAMS_DICT = {}
PARAMS_DICT['v0_full_aux'] = {'LLM_judge' : 0, 'allow_ties' : 0, 'aux_mode' : 'text_with_para_and_noise', 'guidance_mode' : 'none', 'json_params' : {'include_emotion' : 1, 'include_audio_quality' : 1, 'include_aggregate_audio_quality' : 0, 'include_audio_properties' : 1, 'include_speaker_consistency' : 1, 'include_emphasis' : 0, 'include_word_level_audio_properties' : 0}}
PARAMS_DICT['v0_no_aux'] = copy.deepcopy(PARAMS_DICT['v0_full_aux'])
PARAMS_DICT['v0_no_aux']['aux_mode'] = 'none'
PARAMS_DICT['v1a_no_aux'] = copy.deepcopy(PARAMS_DICT['v0_no_aux'])
PARAMS_DICT['v1a_no_aux']['guidance_mode'] = 'semantics'
PARAMS_DICT['v1b_no_aux'] = copy.deepcopy(PARAMS_DICT['v0_no_aux'])
PARAMS_DICT['v1b_no_aux']['guidance_mode'] = 'semantics_and_properties'
PARAMS_DICT['v1c_no_aux'] = copy.deepcopy(PARAMS_DICT['v1b_no_aux'])
PARAMS_DICT['v1c_no_aux']['json_params']['include_audio_quality'] = 0
PARAMS_DICT['v1c_no_aux']['json_params']['include_aggregate_audio_quality'] = 1
PARAMS_DICT['v1d_no_aux'] = copy.deepcopy(PARAMS_DICT['v0_no_aux'])
PARAMS_DICT['v1d_no_aux']['json_params']['include_audio_quality'] = 0
PARAMS_DICT['v1d_no_aux']['json_params']['include_aggregate_audio_quality'] = 1
PARAMS_DICT['v1e_no_aux'] = copy.deepcopy(PARAMS_DICT['v0_no_aux'])
PARAMS_DICT['v1e_no_aux']['guidance_mode'] = 'properties'
PARAMS_DICT['v1f_no_aux'] = copy.deepcopy(PARAMS_DICT['v1d_no_aux'])
PARAMS_DICT['v1f_no_aux']['guidance_mode'] = 'properties'
PARAMS_DICT['v1g_no_aux'] = copy.deepcopy(PARAMS_DICT['v1d_no_aux'])
PARAMS_DICT['v1g_no_aux']['guidance_mode'] = 'semantics'
PARAMS_DICT['v0_LLM_judge'] = copy.deepcopy(PARAMS_DICT['v0_no_aux'])
PARAMS_DICT['v0_LLM_judge']['LLM_judge'] = 1
PARAMS_DICT['v0_LLM_judge_with_ties'] = copy.deepcopy(PARAMS_DICT['v0_LLM_judge'])
PARAMS_DICT['v0_LLM_judge_with_ties']['allow_ties'] = 1
PARAMS_DICT['v2'] = {'LLM_judge' : 0, 'allow_ties' : 0, 'is_v2' : 1, 'give_transcript_hint' : 0, 'json_params' : {'include_emotion' : 1, 'include_accent' : 1, 'include_audio_quality' : 1, 'include_audio_properties' : 1}}
PARAMS_DICT['v2_LLM_judge'] = {'LLM_judge' : 1, 'allow_ties' : 0, 'is_v2' : 1, 'give_transcript_hint' : 0}
PARAMS_DICT['v2_transcript_hint'] = {'LLM_judge' : 0, 'allow_ties' : 0, 'is_v2' : 1, 'give_transcript_hint' : 1, 'json_params' : {'include_emotion' : 1, 'include_accent' : 1, 'include_audio_quality' : 1, 'include_audio_properties' : 1}}
PARAMS_DICT['v2_transcript_hint_LLM_judge'] = {'LLM_judge' : 1, 'allow_ties' : 0, 'is_v2' : 1, 'give_transcript_hint' : 1}


def extract_input_aux(example_info, params):
    assert(False)
    p = params
    input_aux = {}
    assert(p['aux_mode'] in ['none', 'text_only', 'text_with_para', 'text_with_para_and_noise'])
    if p['aux_mode'] == 'text_with_para_and_noise':
        input_aux[p['aux_mode']] = example_info['text']
    elif p['aux_mode'] == 'text_with_para':
        assert(False) #not (yet) implemented
    elif p['aux_mode'] == 'text_only':
        assert(False) #not (yet) implemented
    else:
        assert(p['aux_mode'] == 'none')

    return input_aux


def extract_json_from_audio_v2(audio_path, user_or_agent, params):
    p = params
    assert(user_or_agent in ['user', 'agent'])
    json_dict = {}
    audio_path = audio_path.replace('tongue twisters', 'tongue_twister')
    audio_path = os.path.join('json_as_a_judge/s2sarena_experiments/audio_files', audio_path)
    transcription, word_chunks = whisper_transcribe_v2(audio_path)
    if p['LLM_judge']:
        return transcription
    
    json_dict[user_or_agent + '_text_transcription'] = transcription
    if p['json_params']['include_emotion']:
        emotion_scores = emotion_to_vec_scores(audio_path)
        json_dict[user_or_agent + '_emotion'] = emotion_scores
    
    if p['json_params']['include_accent']:
        accent_scores = get_accent(audio_path)
        json_dict[user_or_agent + '_accent'] = accent_scores
    
    if p['json_params']['include_audio_quality']:
        my_audio_quality_scores = audio_quality_scores(audio_path)
        my_audio_quality_scores.pop('UTMOSv2_Mean_Opinion_Score', None)
        json_dict[user_or_agent + '_audio_quality'] = my_audio_quality_scores
    
    if p['json_params']['include_audio_properties']:
        audio_prop = audio_properties_v2(audio_path, word_chunks)
        json_dict[user_or_agent + '_audio_properties'] = audio_prop

    return json_dict


def extract_json_from_audio(audio_path, user_or_agent, params):
    assert(False)
    p = params
    assert(user_or_agent in ['user', 'agent'])
    json_dict = {}
    audio_path = audio_path.replace('tongue twisters', 'tongue_twister')
    audio_path = os.path.join('json_as_a_judge/s2sarena_experiments/audio_files', audio_path)
    transcription, word_chunks = whisper_transcribe(audio_path)
    if p['LLM_judge']:
        return transcription

    json_dict[user_or_agent + '_text_transcription'] = transcription
    if p['json_params']['include_emotion']:
        emotion_scores = emotion_to_vec_scores(audio_path)
        json_dict[user_or_agent + '_emotion'] = emotion_scores

    if p['json_params']['include_audio_quality'] or p['json_params']['include_aggregate_audio_quality']:
        my_audio_quality_scores = audio_quality_scores(audio_path)
        if p['json_params']['include_audio_quality']:
            json_dict[user_or_agent + '_audio_quality'] = my_audio_quality_scores

        if p['json_params']['include_aggregate_audio_quality']:
            my_aggregate_audio_quality_score = str(np.mean([float(my_audio_quality_scores[k].split()[0]) for k in sorted(my_audio_quality_scores.keys())]))
            json_dict[user_or_agent + '_aggregate_audio_quality'] = my_aggregate_audio_quality_score

    if p['json_params']['include_audio_properties']:
        audio_prop = audio_properties(audio_path, word_chunks)
        json_dict[user_or_agent + '_audio_properties'] = audio_prop

    if p['json_params']['include_speaker_consistency']:
        _, cos_score = get_consistency_score(audio_path)
        json_dict[user_or_agent + '_speaker_consistency'] = cos_score

    #not-yet-implemented things...
    if p['json_params']['include_emphasis']:
        assert(False) #not yet implemented

    if p['json_params']['include_word_level_audio_properties']:
        assert(False) #not yet implemented

    return json_dict


#return input_json, input_aux, outputA_json, outputB_json
def extract_features(example_info, params):
    p = params
    if p['is_v2']:
        input_aux = {}
        input_json = extract_json_from_audio_v2(example_info['input_path'], 'user', p)
        outputA_json = extract_json_from_audio_v2(example_info[example_info['model_a']], 'agent', p)
        outputB_json = extract_json_from_audio_v2(example_info[example_info['model_b']], 'agent', p)
    else:
        input_aux = extract_input_aux(example_info, p)
        input_json = extract_json_from_audio(example_info['input_path'], 'user', p)
        outputA_json = extract_json_from_audio(example_info[example_info['model_a']], 'agent', p)
        outputB_json = extract_json_from_audio(example_info[example_info['model_b']], 'agent', p)
    
    return input_json, input_aux, outputA_json, outputB_json


def get_prompt_template(params, dimensionwise):
    p = params
    if p['is_v2']:
        suffix = ['', '_dimensionwise'][dimensionwise]
        if p['LLM_judge']:
            prompt_template_filename = 'json_as_a_judge/prompts/pairwise_prompt_s2sarena_LLM_judge_v2' + suffix + '.txt'
        else:
            prompt_template_filename = 'json_as_a_judge/prompts/pairwise_prompt_s2sarena_v2' + suffix + '.txt'
    else:
        assert(not dimensionwise)
        if p['LLM_judge']:
            prompt_template_filename = 'json_as_a_judge/prompts/pairwise_prompt_s2sarena_LLM_judge.txt'
        else:
            prompt_template_filename = 'json_as_a_judge/prompts/pairwise_prompt_s2sarena.txt'

    with open(prompt_template_filename, 'r') as f:
        prompt_template = f.read()

    return prompt_template


#only pop top-level keys (for now)
#this means if you want to ablate two things independently, then they have to be two separate top-level entries
def produce_json_description(outputA_json, params):
    p = params
    if p['is_v2']:
        json_description = copy.deepcopy(FULL_JSON_DESCRIPTION_V2)
    else:
        json_description = copy.deepcopy(FULL_JSON_DESCRIPTION)

    top_level_keys = sorted(json_description.keys())
    for k in top_level_keys:
        if k.replace('user/agent_', 'agent_') not in outputA_json:
            json_description.pop(k, None)

    return json.dumps(json_description, indent=2)


def make_prompt(input_json, input_aux, outputA_json, outputB_json, params, dimensionwise):
    p = params
    prompt_template = get_prompt_template(p, dimensionwise)
    prompt_template = Template(prompt_template)
    if not dimensionwise:
        if p['allow_ties']:
            label_description = '"1" if the first audio is better, "2" if the second audio is better, or "tie" if they are equally good/bad.'
        else:
            label_description = '"1" if the first audio is better or "2" if the second audio is better.'

    if not p['is_v2']:
        guidance = GUIDANCE_DICT[p['guidance_mode']]

    transcript_hint = ''
    if p['is_v2'] and dimensionwise and p['give_transcript_hint']:
        transcript_hint = '\n- (Remember, mispronunciations often show up as nonsense words in the response transcript, so be sure to look for those)\n'

    if p['LLM_judge']:
        if p['is_v2']:
            if dimensionwise:
                prompt = prompt_template.substitute(transcript_hint=transcript_hint, input_text_transcription=input_json, outputA_text_transcription=outputA_json, outputB_text_transcription=outputB_json)
            else:
                prompt = prompt_template.substitute(label_description=label_description, input_text_transcription=input_json, outputA_text_transcription=outputA_json, outputB_text_transcription=outputB_json)
        else:
            assert(not dimensionwise)
            prompt = prompt_template.substitute(label_description=label_description, input_text_transcription=input_json, outputA_text_transcription=outputA_json, outputB_text_transcription=outputB_json, guidance=guidance)
        
        return prompt

    json_description = produce_json_description(outputA_json, p)
    if not p['is_v2']:
        input_aux_description1, input_aux_description2 = '', ''
        if p['aux_mode'] == 'text_with_para_and_noise':
            input_aux_description1 = 'In addition to this, the user has provided a ground-truth text transcript of their own speech, which may or may not be annotated with tags describing the user\'s paralinguistics and any background noises. This "auxiliary input" will be provided to you.'
            input_aux_description2 = 'Here is the "auxiliary input" provided by the user:'
        elif p['aux_mode'] == 'text_with_para':
            assert(False) #not (yet) implemented
        elif p['aux_mode'] == 'text_only':
            assert(False) #not (yet) implemented
        else:
            assert(p['aux_mode'] == 'none')

        if p['aux_mode'] == 'none':
            input_aux = ''
        else:
            input_aux = json.dumps(input_aux, indent=2)

    input_json = json.dumps(input_json, indent=2)
    outputA_json = json.dumps(outputA_json, indent=2)
    outputB_json = json.dumps(outputB_json, indent=2)

    if p['is_v2']:
        if dimensionwise:
            prompt = prompt_template.substitute(transcript_hint=transcript_hint, json_description=json_description, input_json=input_json, input_aux=input_aux, outputA_json=outputA_json, outputB_json=outputB_json)
        else:
            prompt = prompt_template.substitute(json_description=json_description, label_description=label_description, input_json=input_json, input_aux=input_aux, outputA_json=outputA_json, outputB_json=outputB_json)
    else:
        prompt = prompt_template.substitute(json_description=json_description, label_description=label_description, input_aux_description1=input_aux_description1, input_aux_description2=input_aux_description2, input_json=input_json, input_aux=input_aux, outputA_json=outputA_json, outputB_json=outputB_json, guidance=guidance)

    return prompt


#probably no actual need for params yet
def run_inference(prompt, params, llm_name, dimensionwise):
    p = params
    pred, full_response = run_llm(prompt, is_json=True, llm_name=llm_name, dimensionwise=dimensionwise)
    if pred is None:
        return None, None

    return pred, full_response


def get_results_dict_filename(params_key, params, llm_name, dimensionwise, rep):
    p = params
    return os.path.join('json_as_a_judge/s2sarena_experiments/results', 'results-%s-%s-dimensionwise%d-rep%d.json'%(params_key, llm_name, dimensionwise, rep))


def s2sarena_json_as_a_judge_inference(params_key, llm_name, dimensionwise, rep):
    print(PARAMS_DICT)
    p = PARAMS_DICT[params_key]
    dimensionwise = int(dimensionwise)
    rep = int(rep)
    assert(llm_name in SUPPORTED_LLM_NAMES)

    example_info_dict = load_example_info_dict()
    print('total of %d examples'%(len(example_info_dict)))
    results_dict = {'params_key' : params_key, 'params' : p, 'outputs' : {}, 'llm_name' : llm_name, 'dimensionwise' : dimensionwise}
    results_dict_filename = get_results_dict_filename(params_key, p, llm_name, dimensionwise, rep)
    if USE_SAVED_PROGRESS and os.path.exists(results_dict_filename):
        with open(results_dict_filename, 'r') as f:
            results_dict = json.load(f)

        print('oh goody! found %d examples already computed!'%(len(results_dict['outputs'])))

    for t, k in tqdm(enumerate(sorted(example_info_dict.keys(), key=int))):
        if USE_SAVED_PROGRESS and k in results_dict['outputs']:
            print('skipping "%s" (because saved!)'%(k))
            continue

        if k in BAD_KEYS:
            print('skipping "%s" (because bad!)'%(k))
            continue

        print('computing "%s"...'%(k))
        example_info = example_info_dict[k]
        if 'input/noise' in example_info['input_path'] or k == '329':
            print('skipping "%s" (because missing!)'%(k))
            continue

        output = {}
        input_json, input_aux, outputA_json, outputB_json = extract_features(example_info, p)
        prompt = make_prompt(input_json, input_aux, outputA_json, outputB_json, p, dimensionwise)
        pred = None
        pred, full_response = run_inference(prompt, p, llm_name, dimensionwise)
        if pred is None: #try again...
            pred, full_response = run_inference(prompt, p, llm_name, dimensionwise)

        if pred is None:
            print('skipping "%s" (because error!)'%(k))
            continue

        output = {'example_info' : example_info, 'input_json' : input_json, 'input_aux' : input_aux, 'outputA_json' : outputA_json, 'outputB_json' : outputB_json, 'prompt' : prompt, 'pred' : pred, 'full_response' : full_response}
        results_dict['outputs'][k] = output
        if t == 1 or (t > 0 and t % 5 == 0):
            with open(results_dict_filename, 'w') as f:
                json.dump(results_dict, f)

    with open(results_dict_filename, 'w') as f:
        json.dump(results_dict, f)


def usage():
    print('Usage: python s2sarena_json_as_a_judge_inference.py <params_key> <llm_name> <dimensionwise> <rep>')


if __name__ == '__main__':
    s2sarena_json_as_a_judge_inference(*(sys.argv[1:]))
