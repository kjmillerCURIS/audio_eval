import os
import sys
import copy
import json
import random
from tqdm import tqdm
from string import Template
import pdb
sys.path.append('.')
from llm_utils import run_llm


LOG_DIR = 'WinoSound/paralinguistic_quality_and_diversity_logs_v4_exclude_bored_and_endimpatienthesitant'
INPUTS = ['776', 'gi', '2395', 'gi', '2754', 'gi', '2848', 'gi', '3903', 'gi', '4473', 'gi', '7593', 'gi', '8306', 'gi', '13112', 'gi', '19933', 'gi', '23278', 'gi', '26509', 'gi', '32820', 'gi', '34102', 'gi', '37879', 'gi', '38997', 'gi', '39299', 'gi', '40415', 'gi', '40867', 'gi', '42036', 'gi']


def print_result(result):
    for turn in result['turns']:
        print('%s: "%s"'%(['HUMAN', 'VA'][turn['is_agent']], turn['text']))

    print('(dataset = %s)'%(result['source_dataset']))


def stringify_result(result):
    lines = []
    for turn in result['turns']:
        lines.append('%s: "%s"'%(['HUMAN', 'VA'][turn['is_agent']], turn['text']))

    return '\n'.join(lines)


def run_llm_generation(example, input_or_output, index):
    if input_or_output == 'input':
        aug_prompt_template_filename = 'WinoSound/prompts/inaug_prompt_emotion_only_v4_exclude_bored_and_endimpatienthesitant.txt'
        refinement_prompt_filename = 'WinoSound/prompts/inaug_prompt_emotion_only_v4_refinement.txt'
        log_filename = os.path.join(LOG_DIR, 'input_aug', '%d.txt'%(index))
    elif input_or_output == 'output':
        assert(False) #FIXME: fix prompt before trying this
        #aug_prompt_template_filename = 'WinoSound/prompts/outaug_prompt_emotion_only.txt'
        #log_filename = os.path.join(LOG_DIR, 'output_aug', '%d.txt'%(index))
    else:
        assert(False)

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(aug_prompt_template_filename, 'r') as f:
        aug_prompt_template = Template(f.read())

    aug_prompt = aug_prompt_template.substitute(source_conversation=example)
    with open(refinement_prompt_filename, 'r') as f:
        refinement_prompt = f.read()

    while True:
        responses = run_llm([aug_prompt, refinement_prompt], llm_name='gemini-2.5-flash', skip_config=True)
        if any([r is None for r in responses]):
            print('ope, got a None response for some reason, let\'s try again!')
        else:
            break

    response = responses[0] + '\n\n=======AFTER REFINEMENT=======\n\n' + responses[1]
    preamble = '=======SOURCE CONVO=======\n\n' + example + '\n\n=======LLM INITIAL RESPONSE=======\n\n'
    with open(log_filename, 'w') as f:
        f.write(preamble + response)

    print(response)


def print_stats(results):
    print('N = %d'%(len(results)))
    dataset_counts = {}
    for result in tqdm(results):
        dataset = result['source_dataset']
        if dataset not in dataset_counts:
            dataset_counts[dataset] = 0

        dataset_counts[dataset] += 1

    print(dataset_counts)


def data_explorer():
    inputs = copy.deepcopy(INPUTS)
    with open('WinoSound/od3_datasets/od3_train.jsonl', 'r') as f:
        json_list = list(f)

    results = []
    for json_str in tqdm(json_list):
        results.append(json.loads(json_str))

    print_stats(results)
    while True:
        print('')
        while True:
            if len(inputs) == 0:
                s = input('enter an index (< %d) or "r" for random:'%(len(results)))
            else:
                s = inputs[0]
                inputs = inputs[1:]
                print('auto-input "%s"'%(s))

            if s == 'r':
                index = random.choice(range(len(results)))
                break
            elif s.isdigit() and int(s) >= 0 and int(s) < len(results):
                index = int(s)
                break
            elif s == 'gi':
                example = stringify_result(result)
                run_llm_generation(example, 'input', index)
            elif s == 'go':
                example = stringify_result(result)
                run_llm_generation(example, 'output', index)

        print('index = %d'%(index))
        result = results[index]
        print_result(result)
        print('index = %d'%(index))


if __name__ == '__main__':
    data_explorer()
