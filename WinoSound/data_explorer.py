import os
import sys
import json
import random
from tqdm import tqdm
from string import Template
import pdb
sys.path.append('.')
from llm_utils import run_llm


LLM_TARGET_EMOTIONS = ['hesitant', 'panicked', 'impatient', 'apologetic/empathetic/reassuring', 'bored', 'angry', 'neutral', 'sad', 'happy']
LOG_DIR = 'WinoSound/paralinguistic_quality_and_diversity_logs'


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
        aug_prompt_template_filename = 'WinoSound/prompts/inaug_prompt_emotion_only.txt'
        log_filename = os.path.join(LOG_DIR, 'input_aug', '%d.txt'%(index))
    elif input_or_output == 'output':
        aug_prompt_template_filename = 'WinoSound/prompts/outaug_prompt_emotion_only.txt'
        log_filename = os.path.join(LOG_DIR, 'output_aug', '%d.txt'%(index))
    else:
        assert(False)

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(aug_prompt_template_filename, 'r') as f:
        aug_prompt_template = Template(f.read())

    aug_prompt = aug_prompt_template.substitute(source_conversation=example, llm_target_emotions=str(LLM_TARGET_EMOTIONS).replace('\'', '"'))
    response = run_llm(aug_prompt, llm_name='gemini-2.5-flash', skip_config=True)
    with open(log_filename, 'w') as f:
        f.write(response)

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
    with open('WinoSound/od3_datasets/od3_train.jsonl', 'r') as f:
        json_list = list(f)

    results = []
    for json_str in tqdm(json_list):
        results.append(json.loads(json_str))

    print_stats(results)
    while True:
        print('')
        while True:
            s = input('enter an index (< %d) or "r" for random:'%(len(results)))
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
        print(result.keys())
        pdb.set_trace()


if __name__ == '__main__':
    data_explorer()
