import os
import sys
import random
from tqdm import tqdm
from llm_utils import run_llm, fill_out_prompt
from setting_concept_taxonomy_utils import obtain_setting_concept_taxonomy_phase


RANDOM_SEED = 42
PROMPT_FILENAMES = ['prompts/knowledge_base_prompt_stateless.txt', 'prompts/knowledge_base_prompt_stateful.txt']
LEN_RANGES = [(100, 200), (100, 200)]


def load_prompts():
    prompts = []
    for prompt_filename in PROMPT_FILENAMES:
        with open(prompt_filename, 'r') as f:
            prompts.append(f.read())

    return prompts


def generate_knowledge_base_one(setting_name, user_name, concept, stateful, prompts):
    len_range_min, len_range_max = LEN_RANGES[stateful]
    query = fill_out_prompt(prompts[stateful], setting_name=setting_name, user_name=user_name, concept=concept, len_range_min=str(len_range_min), len_range_max=str(len_range_max))
    if stateful:
        knowledge_base, _ = run_llm(query, is_json=True, return_json_as_str=True)
    else:
        knowledge_base = run_llm(query, is_json=False)

    return knowledge_base


def generate_knowledge_base():
    random.seed(RANDOM_SEED)
    setting_concept_taxonomy = obtain_setting_concept_taxonomy_phase()
    prompts = load_prompts()
    counters = [0, 0]
    while True:
        while True:
            stateful = input('stateless (0) or stateful (1) (or kill script to exit)?')
            if stateful in ['0', '1']:
                stateful = int(stateful)
                break

        concept = random.choice(setting_concept_taxonomy['leaf_paths'])
        concept = ' >> '.join(concept)
        setting_name, user_name = setting_concept_taxonomy['setting_name'], setting_concept_taxonomy['user_name']
        print(concept)
        print('generating knowledge base...')
        knowledge_base = generate_knowledge_base_one(setting_name, user_name, concept, stateful, prompts)
        print(knowledge_base)
        counter = counters[stateful]
        output_filename = os.path.join('knowledge_bases', 'knowledge_base_%s_stateful%d_%d.%s'%(setting_name.replace(' ', '_'), stateful, counter, ['txt', 'json'][stateful]))
        with open(output_filename, 'w') as f:
            f.write(knowledge_base)

        counters[stateful] = counter + 1


if __name__ == '__main__':
    generate_knowledge_base()
