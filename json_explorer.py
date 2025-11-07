import os
import sys
import json
from tqdm import tqdm


SETTING_NAMES = ['hospital', 'hospital', 'bank', 'indian_restaurant']


def load_experiment_data(setting_index, t):
    json_filename = os.path.join('outputs', '%s%d_%d.json'%(SETTING_NAMES[setting_index], setting_index, t))
    with open(json_filename, 'r') as f:
        experiment_data = json.load(f)

    return experiment_data


def json_explorer(setting_index, t):
    setting_index, t = int(setting_index), int(t)

    experiment_data = load_experiment_data(setting_index, t)
    while True:
        s = input('whaddaya wanna see (k/h/cA/uA/ihA/isA/ehA/esA/tA)?')
        if s == 'k':
            print(experiment_data['knowledge_base_as_str'])
        elif s == 'h':
            print(experiment_data['high_level'])
        elif len(s) >= 2 and s[-1] in ['A', 'B']:
            if s[:-1] == 'c': #conversation
                print(experiment_data['conversation_pair_simplified'][s[-1]])
            elif s[:-1] == 'u': #user utterance
                print(experiment_data['conversation_pair_simplified'][s[-1]]['user'])
            elif s[:-1] == 'ih': #hard instruction
                print(experiment_data['conversation_pair_simplified'][s[-1]]['eval_instructions']['text_eval_instructions']['hard_skill_eval_instructions'])
            elif s[:-1] == 'is': #soft instruction
                print(experiment_data['conversation_pair_simplified'][s[-1]]['eval_instructions']['text_eval_instructions']['soft_skill_eval_instructions'])
            elif s[:-1] == 'eh': #hard evaluation (print one at a time)
                outputs = experiment_data['evaluator_outputs'][s[-1]]['text_eval_outputs']['hard_skills']
                for output in outputs:
                    print(output)
                    print('')

            elif s[:-1] == 'es': #soft evaluation (print one at a time)
                outputs = experiment_data['evaluator_outputs'][s[-1]]['text_eval_outputs']['soft_skills']
                for output in outputs:
                    print(output)
                    print('')

            elif s[:-1] == 't': #transcript
                print(experiment_data['eval_tool_outputs'][s[-1]]['transcriber_outputs']['transcript'])

            print('')

def usage():
    print('Usage: python json_explorer.py <setting_index> <t>')


if __name__ == '__main__':
    json_explorer(*(sys.argv[1:]))
