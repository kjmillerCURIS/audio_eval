import os
import sys
import copy
import glob
import json
from pprint import pprint
import random
import re
from tqdm import tqdm
from setting_concept_taxonomy_utils import obtain_setting_concept_taxonomy_phase
from llm_utils import run_llm, fill_out_prompt
from openai_tts_demo import tts_prompt_hack


RANDOM_SEED = 0
USER_REGISTERS = ["the user is a layperson", "user has above-average expertise", ""]
QUERY_COMPLEXITIES = ["the user's query need not be limited to this concept", "this concept shouldn't be the only concept mentioned", "this concept should be the only concept mentioned", ""]
SHOW_REVISION = True


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


def simplify_conversation_json(conversation):
    simplified_json = copy.deepcopy(conversation)
    for role in ['user', 'agent']:
        simplified_json[role].pop('words', None)
        simplified_json[role].pop('text', None)

    return simplified_json


def is_valid(conversation_pair):
    N = len(conversation_pair['A']['user']['words'])
    if len(conversation_pair['B']['user']['words']) != N:
        return False

    for ab in ['A', 'B']:
        for k in ['text', 'text_with_emphasis']:
            if len(conversation_pair[ab]['user'][k].split()) != N:
                return False

    return True


def convert_emphasis(text_with_emphasis):
    def replacer(match):
        # Split the contents of the emphasis tag into words
        words = match.group(1).split()
        # Capitalize and double-asterisk each word
        emphasized = ' '.join(f'**{word.upper()}**' for word in words)
        return emphasized

    # Replace all <emphasis>...</emphasis> sections
    return re.sub(r'<emphasis>(.*?)</emphasis>', replacer, text_with_emphasis)


def generate_conversation_pair_phase(setting_concept_taxonomy):
    setting_name, user_name = setting_concept_taxonomy['setting_name'], setting_concept_taxonomy['user_name']
    random.seed(RANDOM_SEED)
    with open('conversation_pair_prompt.txt', 'r') as f:
        prompt_template = f.read()

    with open('emphasis_revision_prompt.txt', 'r') as f:
        revision_prompt = f.read()

    with open('conversation_types.json', 'r') as f:
        conversation_types = json.load(f)

    check_conversation_types(conversation_types)
    t = 0
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
        while True:
            print('generating...')
            init_conversation_pair, full_response = run_llm(prompt, is_json=True)
            print('revising...')
            conversation_pair, revision_full_response = run_llm([prompt, full_response, revision_prompt], is_json=True)
            revision_part = revision_full_response.split('{')[0]
            if is_valid(conversation_pair):
                break

            print('let\'s try that again...')

        if SHOW_REVISION:
            print('before revision:')
            for ab in ['A', 'B']:
                simplified_json = simplify_conversation_json(init_conversation_pair[ab])
                print('Conversation %s:'%(ab))
                print(simplified_json)
                print('')

            print('revision explanation:')
            print(revision_part)
            print('')
            print('after_revision:')

        if not SHOW_REVISION:
            print('tts...')

        for ab in ['A', 'B']:
            simplified_json = simplify_conversation_json(conversation_pair[ab])
            print('Conversation %s:'%(ab))
            print(simplified_json)
            print('')
            with open('outputs/%s_%d_%s.json'%(setting_name.replace(' ', '_'), t, ab), 'w') as f:
                json.dump(simplified_json, f)

            for role in ['user', 'agent']:
                audio_filename = 'outputs/%s_%d_%s_%s.wav'%(setting_name.replace(' ', '_'), t, ab, role)
                text_with_emphasis = convert_emphasis(conversation_pair[ab][role]['text_with_emphasis'])
                emotion = conversation_pair[ab][role]['emotion']
                tts_prompt_hack(text_with_emphasis, emotion, audio_filename)

        print('Would you like to generate another? If so, type "c", otherwise kill the script.')
        import pdb
        pdb.set_trace()
        t += 1


def conversation_pair_generator_product_distribution():
    setting_concept_taxonomy = obtain_setting_concept_taxonomy_phase()
    generate_conversation_pair_phase(setting_concept_taxonomy)

if __name__ == '__main__':
    conversation_pair_generator_product_distribution()
