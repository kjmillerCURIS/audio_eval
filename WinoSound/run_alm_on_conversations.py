import os
import sys
import copy
import glob
import json
import random
from tqdm import tqdm
import google.generativeai as genai
from openai import OpenAI
import io
import base64
from pydub import AudioSegment
sys.path.append('.')
from openai_utils import GOOGLE_API_KEY, OPENAI_API_KEY
from extract_json import extract_json
genai.configure(api_key=GOOGLE_API_KEY)
gemini_flash_model = genai.GenerativeModel('models/gemini-2.5-flash')
openai_client = OpenAI(api_key=OPENAI_API_KEY)


RANDOM_SEED = 0
SAVE_FREQ = 5
MAX_NUM_ALM_ATTEMPTS = 5
EXAMPLES_DIR = 'WinoSound/full_conversation_pair_examples/full_conversation_pair_examples-for_website'
RESULTS_DIR = 'WinoSound/full_conversation_pair_results'


#let's just say messages is a list, and we return a string
def call_alm(messages, alm_name):
    if alm_name == 'gemini-2.5-flash':
        try:
            response = gemini_flash_model.generate_content([{'role' : 'user', 'parts' : messages}])
        except Exception as e:
            print(f'got gemini-2.5-flash ALM exception {e}')
            return None

        return (response.text.strip() if hasattr(response, 'text') else None)
    elif alm_name == 'gpt4o':
        try:
            response = openai_client.chat.completions.create(model='gpt-4o-audio-preview', messages=[{'role' : 'user', 'content' : messages}])
        except Exception as e:
            print(f'got gpt4o ALM exception {e}')
            return None

        return response.choices[0].message.content.strip()
    else:
        assert(False)


# ==== Helper: Convert to 16kHz WAV bytes ====
def convert_audio_to_bytes(audio_path: str) -> bytes:
    audio = AudioSegment.from_file(audio_path)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)

    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    return buf.read()


# ==== Helper: base64 encoding ====
def encode_audio_file(file_path: str) -> str:
    """Read an audio file and return base64 string."""
    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

    return encoded_string


def make_text_message(text, alm_name):
    if alm_name == 'gemini-2.5-flash':
        return {'text' : text}
    elif alm_name == 'gpt4o':
        return {'type' : 'text', 'text' : text}
    else:
        assert(False)


def make_audio_message(audio_path, alm_name):
    if alm_name == 'gemini-2.5-flash':
        audio_bytes = convert_audio_to_bytes(audio_path)
        return {"mime_type": "audio/wav", "data": audio_bytes}
    elif alm_name == 'gpt4o':
        audio_bytes = encode_audio_file(audio_path)
        return {"type": "input_audio", "input_audio": {"data": audio_bytes, "format": "wav"}}
    else:
        assert(False)


#CR should be "C" or "R", indicating whether it's convo or response
#name_in_prompt should be something like "A" or "B" or "1" or "2" or ""
def make_prompt_audio_messages(element, CR, name_in_prompt, last_only, alm_name):
    assert(CR in ['C', 'R'])
    assert(name_in_prompt in ['A', 'B', '1', '2', ''])
    speakers = {'HUMAN' : 'human', 'VA' : 'voice assistant'}
    messages = []
    debug_messages = []
    if CR == 'C':
        for t, turn_tuple in enumerate(element):
            if last_only and (t < len(element) - 1):
                continue

            description = 'Conversation C%s turn %d (%s) clip:'%(name_in_prompt, t, speakers[turn_tuple['speaker']])
            audio_path = os.path.join(EXAMPLES_DIR, turn_tuple['audio_path'])
            messages.append(make_text_message(description, alm_name))
            messages.append(make_audio_message(audio_path, alm_name))
            debug_messages.append(description)
            debug_messages.append(audio_path)

        return messages, debug_messages

    elif CR == 'R':
        description = 'Response R%s (%s) clip:'%(name_in_prompt, speakers[element['speaker']])
        audio_path = os.path.join(EXAMPLES_DIR, element['audio_path'])
        messages.append(make_text_message(description, alm_name))
        messages.append(make_audio_message(audio_path, alm_name))
        debug_messages.append(description)
        debug_messages.append(audio_path)
        return messages, debug_messages

    else:
        assert(False)


def make_prompt(challenge_type, para_cue_type, last_only):
    para_cue_part = {'no_para_cue' : '\n', 'soft_para_cue' : 'Focus on the human\'s tone of voice when making your decision.\n', 'hard_para_cue' : 'Focus *only* on the human\'s tone of voice, and not their words, when making your decision.\n'}[para_cue_type]
    if challenge_type == 'pairwise':
        last_only_part = ['', 'Only the last human turn of CA and CB are provided. '][last_only]
        prompt = (
            'You will hear two conversations between a human user and a voice assistant, as a series of audio clips. ' +
            'First, you will hear CA, which is the first conversation *except* for its final voice assistant response. ' +
            'Next, you will hear CB, which is the second conversation *except* for its final voice assistant response. ' +
            last_only_part +
            'Then, you will hear R1 and R2, which are the possible final voice assistant responses.\n' +
            'Your task: match which response is more appropriate for which conversation based on user experience.\n' +
            para_cue_part +
            'Reply with a JSON dict with entries "reasoning", containing your reasoning, and "pred", containing your predicted matching, ' +
            'which itself should be a JSON dict that is either {{"CA" : "R1", "CB" : "R2"}} or {{"CA" : "R2", "CB" : "R1"}}.\n' +
            'Do not include code blocks for the JSON. Respond with the JSON string only.'
        )
        return prompt
    elif challenge_type == 'pointwiseCR':
        last_only_part = ['', 'Only the last human turn of C is provided. '][last_only]
        prompt = (
            'You will hear a conversation between a human user and a voice assistant, as a series of audio clips. ' +
            'First, you will hear C, which is the conversation *except* for the final voice assistant response. ' +
            last_only_part +
            'Then, you will hear R1 and R2, which are possible final voice assistant responses.\n' +
            'Your task: decide which response is more appropriate for the conversation C based on user experience.\n' +
            para_cue_part +
            'Reply with a JSON dict with entries "reasoning", containing your reasoning, and "pred", containing your prediction, ' +
            'which should be either "R1" or "R2".\n' +
            'Do not include code blocks for the JSON. Respond with the JSON string only.'
        )
        return prompt
    elif challenge_type == 'pointwiseRC':
        last_only_part = ['', 'Only the last human turn of C1 and C2 are provided. '][last_only]
        prompt = (
            'You will hear two conversations between a human user and a voice assistant, as a series of audio clips. ' +
            'First, you will hear R, which is the final voice assistant response to *one* of the conversations. ' +
            'Then, you will hear C1 and C2, which are the coversations up to (but not including) that final response. ' +
            last_only_part +
            'Your task: decide which conversation does the response R better fit, based on user experience.\n' +
            para_cue_part +
            'Reply with a JSON dict with entries "reasoning", containing your reasoning, and "pred", containing your prediction, ' +
            'which should be either "C1" or "C2".\n' +
            'Do not include code blocks for the JSON. Respond with the JSON string only.'
        )
        return prompt
    else:
        assert(False)


#must return boolean
def is_valid_response(response, challenge_type):
    if 'pred' not in response:
        return False

    if challenge_type == 'pairwise':
        return (response['pred'] in [{'CA' : 'R1', 'CB' : 'R2'}, {'CA' : 'R2', 'CB' : 'R1'}])
    elif challenge_type == 'pointwiseCR':
        return (response['pred'] in ['R1', 'R2'])
    elif challenge_type == 'pointwiseRC':
        return (response['pred'] in ['C1', 'C2'])
    else:
        assert(False)


#assume pred and gt are already validated
def compute_credit(pred, gt, challenge_type):
    return int(pred == gt)


def alm_loop(messages, alm_name, challenge_type, gt):
    for _ in range(MAX_NUM_ALM_ATTEMPTS):
        response = call_alm(messages, alm_name)
        if response is None:
            print('ALM returned None, retry...')
            continue

        response = extract_json(response)
        if response is None:
            print('JSON parsing error, retry...')
            continue

        if not is_valid_response(response, challenge_type):
            print('pred not valid, retry...')
            continue

        credit = compute_credit(response['pred'], gt, challenge_type)
        return {'response' : response, 'gt' : gt, 'credit' : credit}

    print('out of retries...')
    return None


#AB=="A" means we present RA and call it "R"
#AB=="B" means we present RB and call it "R"
#flip_output==0 means we present CA (and call it "C1") and then CB (and call it "C2")
#flip_output==1 means we present CB (and call it "C1") and then CA (and call it "C2")
#if flip_output==({"A" : 0, "B" : 1}[AB]) then the gt answer is "C1"
#if flip_output!=({"A" : 0, "B" : 1}[AB]) then the gt answer is "C2"
def process_one_example_pointwiseRC(example, AB, para_cue_type, last_only, alm_name):
    assert(AB in ['A', 'B'])
    flip_output = random.choice([0, 1])
    if flip_output == ({'A' : 0, 'B' : 1}[AB]):
        gt = 'C1'
    else:
        gt = 'C2'

    R_element = example['R' + AB]
    C1_element = example['C' + ['A', 'B'][flip_output]]
    C2_element = example['C' + ['A', 'B'][1 - flip_output]]
    R_messages, R_debug = make_prompt_audio_messages(R_element, 'R', '', last_only, alm_name)
    C1_messages, C1_debug = make_prompt_audio_messages(C1_element, 'C', '1', last_only, alm_name)
    C2_messages, C2_debug = make_prompt_audio_messages(C2_element, 'C', '2', last_only, alm_name)
    prompt = make_prompt('pointwiseRC', para_cue_type, last_only)
    prompt_message = make_text_message(prompt, alm_name)
    messages = [prompt_message] + R_messages + C1_messages + C2_messages
    debug = {'R_debug' : R_debug, 'C1_debug' : C1_debug, 'C2_debug' : C2_debug, 'AB' : AB}
    output = alm_loop(messages, alm_name, 'pointwiseRC', gt)
    if output is None:
        return None

    output['flip_output'] = flip_output
    output['debug'] = debug
    return output


#AB=="A" means we present CA and call it "C"
#AB=="B" means we present CB and call it "C"
#flip_output==0 means we present RA (and call it "R1") and then RB (and call it "R2")
#flip_output==1 means we present RB (and call it "R1") and then RA (and call it "R2")
#if flip_output==({"A" : 0, "B" : 1}[AB]) then the gt answer is "R1"
#if flip_output!=({"A" : 0, "B" : 1}[AB]) then the gt answer is "R2"
def process_one_example_pointwiseCR(example, AB, para_cue_type, last_only, alm_name):
    assert(AB in ['A', 'B'])
    flip_output = random.choice([0, 1])
    if flip_output == ({'A' : 0, 'B' : 1}[AB]):
        gt = 'R1'
    else:
        gt = 'R2'

    C_element = example['C' + AB]
    R1_element = example['R' + ['A', 'B'][flip_output]]
    R2_element = example['R' + ['A', 'B'][1 - flip_output]]
    C_messages, C_debug = make_prompt_audio_messages(C_element, 'C', '', last_only, alm_name)
    R1_messages, R1_debug = make_prompt_audio_messages(R1_element, 'R', '1', last_only, alm_name)
    R2_messages, R2_debug = make_prompt_audio_messages(R2_element, 'R', '2', last_only, alm_name)
    prompt = make_prompt('pointwiseCR', para_cue_type, last_only)
    prompt_message = make_text_message(prompt, alm_name)
    messages = [prompt_message] + C_messages + R1_messages + R2_messages
    debug = {'C_debug' : C_debug, 'R1_debug' : R1_debug, 'R2_debug' : R2_debug, 'AB' : AB}
    output = alm_loop(messages, alm_name, 'pointwiseCR', gt)
    if output is None:
        return None

    output['flip_output'] = flip_output
    output['debug'] = debug
    return output


#flip_input==0 means we present CA (and call it "CA") and then CB (and call it "CB")
#flip_input==1 means we present CB (and call it "CA") and then CA (and call it "CB")
#flip_output==0 means we present RA (and call it "R1") and then RB (and call it "R2")
#flip_output==1 means we present RB (and call it "R1") and then RA (and call it "R2")
#if flip_input==flip_output then the gt answer (from the ALM) is {"CA" : "R1", "CB" : "R2"} 
#if flip_input!=flip_output then the gt answer (from the ALM) is {"CA" : "R2", "CB" : "R1"} 
def process_one_example_pairwise(example, para_cue_type, last_only, alm_name):
    flip_input = random.choice([0, 1])
    flip_output = random.choice([0, 1])
    if flip_input == flip_output:
        gt = {'CA' : 'R1', 'CB' : 'R2'}
    else:
        gt = {'CA' : 'R2', 'CB' : 'R1'}
    CA_element = example['C' + ['A', 'B'][flip_input]]
    CB_element = example['C' + ['A', 'B'][1 - flip_input]]
    R1_element = example['R' + ['A', 'B'][flip_output]]
    R2_element = example['R' + ['A', 'B'][1 - flip_output]]
    CA_messages, CA_debug = make_prompt_audio_messages(CA_element, 'C', 'A', last_only, alm_name)
    CB_messages, CB_debug = make_prompt_audio_messages(CB_element, 'C', 'B', last_only, alm_name)
    R1_messages, R1_debug = make_prompt_audio_messages(R1_element, 'R', '1', last_only, alm_name)
    R2_messages, R2_debug = make_prompt_audio_messages(R2_element, 'R', '2', last_only, alm_name)
    prompt = make_prompt('pairwise', para_cue_type, last_only)
    prompt_message = make_text_message(prompt, alm_name)
    messages = [prompt_message] + CA_messages + CB_messages + R1_messages + R2_messages
    debug = {'CA_debug' : CA_debug, 'CB_debug' : CB_debug, 'R1_debug' : R1_debug, 'R2_debug' : R2_debug}
    output = alm_loop(messages, alm_name, 'pairwise', gt)
    if output is None:
        return None

    output['flip_input'] = flip_input
    output['flip_output'] = flip_output
    output['debug'] = debug
    return output


def process_one_example(example, challenge_type, para_cue_type, last_only, alm_name):
    if challenge_type == 'pairwise':
        return process_one_example_pairwise(example, para_cue_type, last_only, alm_name)
    elif challenge_type in ['pointwiseCR', 'pointwiseRC']:
        outputs = []
        for AB in ['A', 'B']:
            if challenge_type == 'pointwiseCR':
                output = process_one_example_pointwiseCR(example, AB, para_cue_type, last_only, alm_name)
            elif challenge_type == 'pointwiseRC':
                output = process_one_example_pointwiseRC(example, AB, para_cue_type, last_only, alm_name)
            else:
                assert(False)

            if output is None:
                return None

            outputs.append(output)

        return outputs

    else:
        assert(False)


def get_results_filename(challenge_type, para_cue_type, last_only, alm_name):
    return os.path.join(RESULTS_DIR, 'results-conversations-%s-%s-last_only%d-%s.json'%(challenge_type, para_cue_type, last_only, alm_name))


def load_examples():
    info_filenames = sorted(glob.glob(os.path.join(EXAMPLES_DIR, '*/*-info.json')))
    examples = {}
    for info_filename in tqdm(info_filenames):
        k = os.path.basename(os.path.dirname(info_filename))
        assert(k not in examples)
        with open(info_filename, 'r') as f:
            examples[k] = json.load(f)

    return examples


def compute_correct_and_total(results):
    #typical_order means the gt answer is {(CA, R1), (CB, R2)} or (C, R1) or (R, CA)
    #atypical_order is the other gt - we think the ALM might be biased towards the typical order
    total = {'all' : 0, 'typical_order' : 0, 'atypical_order' : 0}
    correct = {'all' : 0, 'typical_order' : 0, 'atypical_order' : 0}
    for k in tqdm(sorted(results['outputs'].keys())):
        output = results['outputs'][k]
        if isinstance(output, list):
            for i, o in enumerate(output):
                correct['all'] += o['credit']
                total['all'] += 1
                if o['flip_output'] == i:
                    correct['typical_order'] += o['credit']
                    total['typical_order'] += 1
                else:
                    correct['atypical_order'] += o['credit']
                    total['atypical_order'] += 1

        else:
            correct['all'] += output['credit']
            total['all'] += 1
            if output['flip_output'] == output['flip_input']:
                correct['typical_order'] += output['credit']
                total['typical_order'] += 1
            else:
                correct['atypical_order'] += output['credit']
                total['atypical_order'] += 1

    return correct, total


def print_stats(results):
    correct, total = compute_correct_and_total(results)
    for m in ['typical_order', 'atypical_order', 'all']:
        print('accuracy(%s) = %.1f (%d / %d)'%(m, 100.0 * correct[m] / max(total[m], 1e-5), correct[m], total[m]))


#challenge_type should be "pairwise", "pointwiseCR", or "pointwiseRC"
#para_cue_type should be "no_para_cue", "soft_para_cue" (listen to para), or "hard_para_cue" (listen to para and ignore content)
#last_only should be 0 or 1, decides if we only give the last turn of the convo (because I suspect that is sufficient)
#alm_name should be the name of the judge model
def run_alm_on_conversations(challenge_type, para_cue_type, last_only, alm_name):
    last_only = int(last_only)

    assert(challenge_type in ['pairwise', 'pointwiseCR', 'pointwiseRC'])
    assert(para_cue_type in ['no_para_cue', 'soft_para_cue', 'hard_para_cue'])
    assert(alm_name in ['gpt4o', 'gemini-2.5-flash'])

    random.seed(RANDOM_SEED)
    examples = load_examples()
    results_filename = get_results_filename(challenge_type, para_cue_type, last_only, alm_name)
    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    results = {'challenge_type' : challenge_type, 'para_cue_type' : para_cue_type, 'last_only' : last_only, 'alm_name' : alm_name, 'outputs' : {}}
    if os.path.exists(results_filename):
        with open(results_filename, 'r') as f:
            results = json.load(f)

    my_keys = [k for k in sorted(examples.keys()) if k not in results['outputs']]
    print('%d total examples, %d already processed, %d to process...'%(len(examples), len(results['outputs']), len(my_keys)))
    for t, k in tqdm(enumerate(my_keys)):
        print('processing "%s"...'%(k))
        output = process_one_example(examples[k], challenge_type, para_cue_type, last_only, alm_name)
        if output is None:
            print('skipping "%s" (could not process)'%(k))
            continue

        results['outputs'][k] = output
        if (t + 1) % SAVE_FREQ == 0:
            with open(results_filename, 'w') as f:
                json.dump(results, f)

            print_stats(results)

    with open(results_filename, 'w') as f:
        json.dump(results, f)

    print_stats(results)


def usage():
    print('Usage: python run_alm_on_conversations.py <challenge_type> <para_cue_type> <last_only> <alm_name>')


if __name__ == '__main__':
    run_alm_on_conversations(*(sys.argv[1:]))
