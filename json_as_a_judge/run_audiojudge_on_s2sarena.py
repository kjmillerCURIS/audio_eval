import os
import sys
import json
import shutil
from tqdm import tqdm
from dotenv import load_dotenv
sys.path.append('.')
from openai_utils import OPENAI_API_KEY, GOOGLE_API_KEY
#os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
from llm_utils import extract_json
from json_as_a_judge.AudioJudge.src.audiojudge import AudioJudge, AudioExample
from json_as_a_judge.AudioJudge.examples.system_prompts import SYSTEM_PROMPTS
from json_as_a_judge.s2sarena_utils import load_example_info_dict


DIMENSIONWISE_SYSTEM_PROMPT_NO_ICL = """
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

DIMENSIONWISE_USER_PROMPT = """
Respond ONLY in text and output valid JSON with keys "reasoning", "content", "voice_quality", and "instruction_following_audio":
"""


THEIR_DATASET_NAME = 'speakbench508'
THEIR_NUM_SHOTS = 4
USE_SAVED_PROGRESS = True
#BAD_KEYS = ['306', '351', '359', '400', '409', '446']
BAD_KEYS = []


def load_ICL_examples():
    with open("json_as_a_judge/AudioJudge/experiments/main_experiments/few_shots_examples.json", "r") as f:
        few_shots_examples = json.load(f)

    few_shots_examples = few_shots_examples[THEIR_DATASET_NAME][:THEIR_NUM_SHOTS]
    examples = []
    for example in few_shots_examples:
        examples.append(
            AudioExample(
                audio1_path=os.path.join(
                    "json_as_a_judge", "AudioJudge", "experiments", "main_experiments", example["audio1_path"]
                ),
                audio2_path=os.path.join(
                    "json_as_a_judge", "AudioJudge", "experiments", "main_experiments", example["audio2_path"]
                ),
                instruction_path=os.path.join(
                    "json_as_a_judge", "AudioJudge", "experiments", "main_experiments", example["instruction_path"]
                ),
                output=json.dumps(example["assistant_message"]),
            )
        )

    return examples


def get_user_prompt(dataset_name: str) -> str:
    user_prompt = ""
    if "speakbench" in dataset_name:
        user_prompt = (
            "Please analyze which of the two recordings follows the instruction better, or tie. "
            "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
        )
    elif "chatbotarena" in dataset_name:
        user_prompt = (
            "Please analyze which of the two recordings follows the instruction better, or tie, in terms of content of the responses. "
            "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
        )
    return user_prompt


def setup_tools(use_icl, dimensionwise):
    audio_judge = AudioJudge()
    assert(not (use_icl and dimensionwise))
    if dimensionwise:
        user_prompt = DIMENSIONWISE_USER_PROMPT
        system_prompt = DIMENSIONWISE_SYSTEM_PROMPT_NO_ICL
    else:
        user_prompt = get_user_prompt(THEIR_DATASET_NAME)
        system_prompt = SYSTEM_PROMPTS[THEIR_DATASET_NAME]["standard_cot"]

    tools = {'audio_judge' : audio_judge, 'user_prompt' : user_prompt, 'system_prompt' : system_prompt}
    if use_icl:
        tools['examples'] = load_ICL_examples()

    return tools


def one_audiojudge_call(audio1_path, audio2_path, input_path, alm_name, use_icl, dimensionwise, tools):
    examples = None
    concatenation_method = 'no_concatenation'
    assert(not (use_icl and dimensionwise))
    if use_icl:
        examples = tools['examples']
        concatenation_method = 'examples_and_test_concatenation'

    already_tried = False
    while True:
        response = tools['audio_judge'].judge_audio(
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    instruction_path=input_path,
                    user_prompt=tools['user_prompt'],
                    system_prompt=tools['system_prompt'],
                    model=alm_name,
                    examples=examples,
                    concatenation_method=concatenation_method,
                )

        d = extract_json(response['response'])
        if d is not None: #success!
            return response
        else: #cleanup and...
            tools['audio_judge'].clear_cache()
            if already_tried: #give up!
                return None
            else: #retry!
                already_tried = True


#return a pair of responses (one for each ordering)
def run_audiojudge_on_s2sarena_one_example(example_info, tools, alm_name, use_icl, dimensionwise):
    audio_dir = 'json_as_a_judge/s2sarena_experiments/audio_files'
    input_path = os.path.join(audio_dir, example_info['input_path'].replace('input/', 'input_converted/'))
    audio1_path = os.path.join(audio_dir, example_info[example_info['model_a']].replace('output/', 'output_converted/'))
    audio2_path = os.path.join(audio_dir, example_info[example_info['model_b']].replace('output/', 'output_converted/'))
    input_path = input_path.replace('tongue twisters', 'tongue_twister')
    audio1_path = audio1_path.replace('tongue twisters', 'tongue_twister')
    audio2_path = audio2_path.replace('tongue twisters', 'tongue_twister')

    response12 = one_audiojudge_call(audio1_path, audio2_path, input_path, alm_name, use_icl, dimensionwise, tools)
    if response12 is None:
        return None, None

    response21 = one_audiojudge_call(audio2_path, audio1_path, input_path, alm_name, use_icl, dimensionwise, tools)
    if response21 is None:
        return None, None

    return response12, response21


def get_results_dict_filename(alm_name, use_icl, dimensionwise):
    return os.path.join('json_as_a_judge/s2sarena_experiments/audiojudge_results', 'audiojudge_results_%s_icl%d_dimensionwise%d.json'%(alm_name, use_icl, dimensionwise))


def clean_results_dict(results_dict):
    my_keys = sorted(results_dict['outputs'].keys(), key=int)
    for k in my_keys:
        if isinstance(results_dict['outputs'][k]['response12']['response'], str):
            try:
                results_dict['outputs'][k]['response12']['response'] = json.loads(results_dict['outputs'][k]['response12']['response'])
                results_dict['outputs'][k]['response21']['response'] = json.loads(results_dict['outputs'][k]['response21']['response'])
            except:
                print('pop "%s"'%(k))
                results_dict['outputs'].pop(k, None)

    return results_dict


def run_audiojudge_on_s2sarena(alm_name, use_icl, dimensionwise):
    use_icl = int(use_icl)
    dimensionwise = int(dimensionwise)

    load_dotenv()
    tools = setup_tools(use_icl, dimensionwise)
    example_info_dict = load_example_info_dict()
    print('total of %d examples'%(len(example_info_dict)))
    results_dict = {'alm_name' : alm_name, 'outputs' : {}, 'use_icl' : use_icl, 'dimensionwise' : dimensionwise}
    results_dict_filename = get_results_dict_filename(alm_name, use_icl, dimensionwise)
    if USE_SAVED_PROGRESS and os.path.exists(results_dict_filename):
        with open(results_dict_filename, 'r') as f:
            results_dict = json.load(f)

        results_dict = clean_results_dict(results_dict)
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

        response12, response21 = run_audiojudge_on_s2sarena_one_example(example_info, tools, alm_name, use_icl, dimensionwise)
        if response12 is None:
            print('skipping "%s" (because error!)'%(k))
            continue

        output = {'example_info' : example_info, 'response12' : response12, 'response21' : response21}
        results_dict['outputs'][k] = output
        results_dict['outputs'][k]['response12']['response'] = extract_json(results_dict['outputs'][k]['response12']['response'])
        results_dict['outputs'][k]['response21']['response'] = extract_json(results_dict['outputs'][k]['response21']['response'])
        if t == 1 or (t > 0 and t % 5 == 0):
            with open(results_dict_filename, 'w') as f:
                json.dump(results_dict, f)

    with open(results_dict_filename, 'w') as f:
        json.dump(results_dict, f)


def usage():
    print('Usage: python run_audiojudge_on_s2sarena.py <alm_name> <use_icl> <dimensionwise>')


if __name__ == '__main__':
    run_audiojudge_on_s2sarena(*(sys.argv[1:]))
