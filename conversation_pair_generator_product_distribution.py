import os
import sys
import glob
import json
from pprint import pprint
import random
import re
from tqdm import tqdm
from setting_concept_taxonomy_utils import build_setting_concept_taxonomy
from llm_utils import run_llm


RANDOM_SEED = 0
USER_REGISTERS = ["the user is a layperson", "user has above-average expertise", ""]
QUERY_COMPLEXITIES = ["the user's query need not be limited to this concept", "this concept shouldn't be the only concept mentioned", "this concept should be the only concept mentioned", ""]


def sample_conversation_type_pair(conversation_types):
    cA = random.choice(conversation_types['conversation_types'])
    should_remove = []
    if cA not in conversation_types['can_double_sample']:
        should_remove.append(cA)

    for g in conversation_types['do_not_cooccur']:
        if cA in g:
            for c in g:
                if not (c == cA and cA in conversation_types['can_double_sample']):
                    should_remove.append(c)

    remaining = [c for c in conversation_types['conversation_types'] if c not in should_remove]
    cB = random.choice(remaining)
    if random.choice([0,1]):
        cA, cB = cB, cA

    return cA, cB


def check_conversation_types(conversation_types):
    for g in conversation_types['do_not_cooccur']:
        assert(all([c in conversation_types['conversation_types'] for c in g]))

    assert(all([c in conversation_types['conversation_types'] for c in conversation_types['can_double_sample']]))


def obtain_setting_concept_taxonomy_phase():
    setting_concept_taxonomy_filenames = sorted(glob.glob('setting_concept_taxonomies/setting_*.json'))
    setting_concept_taxonomies = {}
    print('loading existing setting concept taxonomies...')
    for filename in tqdm(setting_concept_taxonomy_filenames):
        with open(filename, 'r') as f:
            taxonomy = json.load(f)

        index = int(os.path.splitext(os.path.basename(filename))[0].split('_')[-1])
        assert(index not in setting_concept_taxonomies)
        setting_concept_taxonomies[index] = taxonomy

    while True:
        print('Here are the available taxonomies:')
        for index in sorted(setting_concept_taxonomies.keys()):
            print('%d ==> setting_name="%s", user_name="%s"'%(index, setting_concept_taxonomies[index]['setting_name'], setting_concept_taxonomies[index]['user_name']))

        while True:
            my_input = input('Please enter an index of an existing taxonomy, or "+" to build a new one:')
            if my_input == '+' or (my_input.isdigit() and int(my_input) in setting_concept_taxonomies):
                break

        if my_input == '+':
            new_index = max([index for index in sorted(setting_concept_taxonomies.keys())]) + 1 if len(setting_concept_taxonomies) > 0 else 0
            setting_name = input('please enter setting name (e.g. hospital, bank):')
            user_name = input('please enter user name (e.g. patient, customer):')
            taxonomy = build_setting_concept_taxonomy(setting_name, user_name)
            setting_concept_taxonomies[new_index] = taxonomy
            with open(os.path.join('setting_concept_taxonomies', 'setting_%d.json'%(new_index)), 'w') as f:
                json.dump(taxonomy, f, indent=4, sort_keys=True)
        else:
            return setting_concept_taxonomies[int(my_input)]


    assert(False) #KEVIN


def fill_out_prompt(prompt_template, **kwargs):
    prompt = prompt_template
    for k in sorted(kwargs.keys()):
        prompt = prompt.replace('FORMAT_TARGET' + k + 'FORMAT_TARGET', kwargs[k])

    return prompt


def generate_conversation_pair_phase(setting_concept_taxonomy):
    setting_name, user_name = setting_concept_taxonomy['setting_name'], setting_concept_taxonomy['user_name']
    random.seed(RANDOM_SEED)
    with open('conversation_pair_prompt.txt', 'r') as f:
        prompt_template = f.read()

    with open('conversation_types.json', 'r') as f:
        conversation_types = json.load(f)

    check_conversation_types(conversation_types)
    while True:
        cA, cB = sample_conversation_type_pair(conversation_types)
        setting_concept = random.choice(setting_concept_taxonomy['leaf_paths'] + setting_concept_taxonomy['inner_paths'])
        setting_concept = ' >> '.join(setting_concept)
        user_register = random.choice(USER_REGISTERS)
        query_complexity = random.choice(QUERY_COMPLEXITIES)
        print('setting_name = "%s"'%(setting_name))
        print('user_name = "%s"'%(user_name))
        print('cA = "%s"'%(cA))
        print('cB = "%s"'%(cB))
        print('setting_concept = "%s"'%(setting_concept))
        print('query_complexity = "%s"'%(query_complexity))
        print('user_register = "%s"'%(user_register))
        prompt = fill_out_prompt(prompt_template, setting_name=setting_name, user_name=user_name, cA=cA, cB=cB, setting_concept=setting_concept, query_complexity=query_complexity, user_register=user_register)
        print('generating...')
        conversation_pair = run_llm(prompt, is_json=True)
        print('')
        print('Conversation A:')
        print(conversation_pair['A'])
        print('')
        print('Conversation B:')
        print(conversation_pair['B'])
        print('')
        print('Would you like to generate another? If so, type "c", otherwise kill the script.')
        import pdb
        pdb.set_trace()


def conversation_pair_generator_product_distribution():
    setting_concept_taxonomy = obtain_setting_concept_taxonomy_phase()
    generate_conversation_pair_phase(setting_concept_taxonomy)

if __name__ == '__main__':
    conversation_pair_generator_product_distribution()
