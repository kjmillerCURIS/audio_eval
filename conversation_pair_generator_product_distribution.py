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
from generate_knowledge_base import generate_knowledge_base_one
from generate_knowledge_base import load_prompts as load_knowledge_base_prompts
from llm_utils import run_llm, fill_out_prompt
from openai_tts_demo import tts_prompt_hack
from run_qwen import setup_qwen_model
from run_qwen import run_qwen as _run_qwen
from eval_utils import run_eval_tools as _run_eval_tools
from eval_utils import run_evaluator as _run_evaluator
from eval_utils import setup_eval_tool_models


RANDOM_SEED = None #0
ANSWERABILITIES = ['The user query should be answerable using the information in the knowledge base excerpt.',
                    'The user query should *not* be answerable using the information in the knowledge base excerpt.',
                    '']
USER_REGISTERS = ["the user is a layperson", "user has above-average expertise", ""]
QUERY_COMPLEXITIES = ["the user's query need not be limited to this concept", "this concept shouldn't be the only concept mentioned", "this concept should be the only concept mentioned", ""]
SHOW_REVISION = False


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


#FIXME: include simplified jsons in experiment_data
def generate_conversation_pair_phase(setting_concept_taxonomy, setting_index, t):
    setting_name, user_name = setting_concept_taxonomy['setting_name'], setting_concept_taxonomy['user_name']
    with open('prompts/conversation_pair_prompt.txt', 'r') as f:
        prompt_template = f.read()

    with open('prompts/emphasis_revision_prompt.txt', 'r') as f:
        revision_prompt = f.read()

    with open('conversation_types.json', 'r') as f:
        conversation_types = json.load(f)

    experiment_data = {'bookkeeping' : {'setting_index' : setting_index, 'setting_name' : setting_name, 'user_name' : user_name, 't' : t}}

    check_conversation_types(conversation_types)
    while True:
        cA, cB = sample_conversation_type_pair(conversation_types)
        setting_concept = random.choice(setting_concept_taxonomy['leaf_paths'])
        setting_concept = ' >> '.join(setting_concept)
        user_register = random.choice(USER_REGISTERS)
        query_complexity = random.choice(QUERY_COMPLEXITIES)
        answerability = random.choice(ANSWERABILITIES)
        stateful = random.choice([0, 1])
        print('setting_name = "%s"'%(setting_name))
        print('user_name = "%s"'%(user_name))
        print('cA = "%s"'%(cA))
        print('cB = "%s"'%(cB))
        print('setting_concept = "%s"'%(setting_concept))
        print('query_complexity = "%s"'%(query_complexity))
        print('user_register = "%s"'%(user_register))
        print('answerability = "%s"'%(answerability))
        print('stateful = %d'%(stateful))
        while True:
            yn = input('are you happy with these (y/n)? If yes, we\'ll move on to generation, else we\'ll resample.')
            if yn in ['y', 'n']:
                break

        if yn == 'y':
            break

    experiment_data['high_level'] = {'cA' : cA, 'cB' : cB, 'setting_concept' : setting_concept, 'user_register' : user_register, 'query_complexity' : query_complexity, 'answerability' : answerability, 'stateful' : stateful}
    print('generating knowledge base...')
    knowledge_base = generate_knowledge_base_one(setting_name, user_name, setting_concept, stateful, load_knowledge_base_prompts())
    experiment_data['knowledge_base_as_str'] = knowledge_base
    prompt = fill_out_prompt(prompt_template, setting_name=setting_name, user_name=user_name, cA=cA, cB=cB, setting_concept=setting_concept, query_complexity=query_complexity, user_register=user_register, answerability=answerability, knowledge_base=knowledge_base)
    while True:
        print('generating conversation...')
        init_conversation_pair, full_response = run_llm(prompt, is_json=True)
        print('revising...')
        conversation_pair, revision_full_response = run_llm([prompt, full_response, revision_prompt], is_json=True)
        revision_part = revision_full_response.split('{')[0]
        if is_valid(conversation_pair):
            break

        print('let\'s try that again...')

    experiment_data['init_conversation_pair'] = init_conversation_pair
    experiment_data['revision_part'] = revision_part
    experiment_data['conversation_pair'] = conversation_pair

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

    experiment_data['bookkeeping']['audio_files'] = {}
    experiment_data['conversation_pair_simplified'] = {}
    for ab in ['A', 'B']:
        experiment_data['bookkeeping']['audio_files'][ab] = {}
        simplified_json = simplify_conversation_json(conversation_pair[ab])
        experiment_data['conversation_pair_simplified'][ab] = simplified_json
        print('Conversation %s:'%(ab))
        print(simplified_json)
        print('')
#        with open('outputs/%s_%d_%s.json'%(setting_name.replace(' ', '_'), t, ab), 'w') as f:
#            json.dump(simplified_json, f)

        for role in ['user', 'agent']:
            audio_filename = 'outputs/%s%d_%d_%s_%s.wav'%(setting_name.replace(' ', '_'), setting_index, t, ab, role)
            experiment_data['bookkeeping']['audio_files'][ab][role] = audio_filename
            text_with_emphasis = convert_emphasis(conversation_pair[ab][role]['text_with_emphasis'])
            emotion = conversation_pair[ab][role]['emotion']
            tts_prompt_hack(text_with_emphasis, emotion, audio_filename)

    return experiment_data


#FIXME: obtain Qwen's text from _run_qwen() call, save it in experiment_data
def run_qwen(experiment_data, qwen_model):
    for ab in ['A', 'B']:
        print('running qwen for conversation %s...'%(ab))
        audio_input_filename = experiment_data['bookkeeping']['audio_files'][ab]['user']
        audio_output_filename = os.path.join(os.path.dirname(audio_input_filename), os.path.basename(audio_input_filename).replace('user', 'qwen'))
        assert(audio_input_filename != audio_output_filename)
        setting_name, user_name = experiment_data['bookkeeping']['setting_name'], experiment_data['bookkeeping']['user_name']
        knowledge_base = experiment_data['knowledge_base_as_str']
        _run_qwen(audio_input_filename, setting_name, user_name, knowledge_base, qwen_model, audio_output_filename)
        experiment_data['bookkeeping']['audio_files'][ab]['qwen'] = audio_output_filename

    return experiment_data


def run_eval_tools(experiment_data, eval_tool_models):
    experiment_data['eval_tool_outputs'] = {}
    for ab in ['A', 'B']:
        print('eval tools for conversation %s...'%(ab))
        audio_path = experiment_data['bookkeeping']['audio_files'][ab]['qwen']
        experiment_data['eval_tool_outputs'][ab] = _run_eval_tools(audio_path, eval_tool_models)

    return experiment_data


def run_evaluator(experiment_data):
    experiment_data['evaluator_outputs'] = {}
    for ab in ['A', 'B']:
        print('evaluator for conversation %s...'%(ab))
        eval_tool_outputs, eval_instructions = experiment_data['eval_tool_outputs'][ab], experiment_data['conversation_pair'][ab]['eval_instructions']
        user_utterance = experiment_data['conversation_pair'][ab]['user']
        user_utterance = {'text_with_emphasis' : user_utterance['text_with_emphasis'], 'emotion' : user_utterance['emotion']}
        user_utterance = json.dumps(user_utterance)
        knowledge_base = experiment_data['knowledge_base_as_str']
        setting_name, user_name = experiment_data['bookkeeping']['setting_name'], experiment_data['bookkeeping']['user_name']
        experiment_data['evaluator_outputs'][ab] = _run_evaluator(eval_tool_outputs, eval_instructions, user_utterance, knowledge_base, setting_name, user_name)

    return experiment_data


def save_experiment_data(experiment_data):
    setting_name, setting_index, t = experiment_data['bookkeeping']['setting_name'], experiment_data['bookkeeping']['setting_index'], experiment_data['bookkeeping']['t']
    experiment_filename = 'outputs/%s%d_%d.json'%(setting_name.replace(' ', '_'), setting_index, t)
    with open(experiment_filename, 'w') as f:
        json.dump(experiment_data, f)


def conversation_pair_generator_product_distribution():
    random.seed(RANDOM_SEED)
    print('loading qwen model...')
    qwen_model = setup_qwen_model()
    print('loading eval tool models...')
    eval_tool_models = setup_eval_tool_models()
    print('all models loaded!')
    setting_concept_taxonomy, setting_index = obtain_setting_concept_taxonomy_phase()
    t = 0
    while True:
        experiment_data = generate_conversation_pair_phase(setting_concept_taxonomy, setting_index, t)
        experiment_data = run_qwen(experiment_data, qwen_model) #local wrapper
        experiment_data = run_eval_tools(experiment_data, eval_tool_models) #just transcriber for now
        experiment_data = run_evaluator(experiment_data)
        save_experiment_data(experiment_data)
        print('Would you like to generate another? If so, type "c", otherwise kill the script.')
        import pdb
        pdb.set_trace()
        t += 1


if __name__ == '__main__':
    conversation_pair_generator_product_distribution()
