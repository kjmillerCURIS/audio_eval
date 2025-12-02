import os
import sys
import copy
import json
import random
from tqdm import tqdm
sys.path.append('.')
from run_alm_on_conversations import get_results_filename, load_examples


def interpret_kk(kk):
    alm_name, para_cue_type, last_only = kk
    return alm_name + ', ' + {'no_para_cue' : 'no paralinguistic cue', 'soft_para_cue' : '"soft" paralinguistic cue', 'hard_para_cue' : '"hard" paralinguistic cue'}[para_cue_type] + ', ' + ['full convo', 'last turn only'][last_only]


def kk_val(kk):
    alm_name, para_cue_type, last_only = kk
    a = {'gemini-2.5-flash' : 0, 'gpt4o' : 1}[alm_name]
    b = {'no_para_cue' : 0, 'soft_para_cue' : 1, 'hard_para_cue' : 2}[para_cue_type]
    c = last_only
    return (a, b, c)


def strip_turn(turn):
    turn = copy.deepcopy(turn)
    turn.pop('display_text')
    return turn


def print_example_result(challenge_type, k, examples, results_dict):
    print(str(k) + ':')
    print('')
    if challenge_type != 'pairwise':
        k, i = k

    #print example
    output = results_dict[sorted(results_dict.keys())[0]]['outputs'][k]
    if challenge_type != 'pairwise':
        output = output[i]

    gt = output['gt']
    example = examples[k]
    if challenge_type == 'pairwise':
        CA = example['C' + ['A', 'B'][output['flip_input']]]
        CB = example['C' + ['B', 'A'][output['flip_input']]]
        R1 = example['R' + ['A', 'B'][output['flip_output']]]
        R2 = example['R' + ['B', 'A'][output['flip_output']]]
        print('Judge\'s "CA":')
        for turn in CA:
            print(strip_turn(turn))

        print('\nJudge\'s "CB":')
        for turn in CB:
            print(strip_turn(turn))

        print('\nJudge\'s "R1":')
        print(strip_turn(R1))
        print('\nJudge\'s "R2":')
        print(strip_turn(R2))
        print('')
    elif challenge_type == 'pointwiseCR':
        C = example['C' + ['A', 'B'][i]]
        R1 = example['R' + ['A', 'B'][output['flip_output']]]
        R2 = example['R' + ['B', 'A'][output['flip_output']]]
        print('Judge\'s "C":')
        for turn in C:
            print(strip_turn(turn))

        print('\nJudge\'s "R1":')
        print(strip_turn(R1))
        print('\nJudge\'s "R2":')
        print(strip_turn(R2))
        print('')
    elif challenge_type == 'pointwiseRC':
        R = example['R' + ['A', 'B'][i]]
        C1 = example['C' + ['A', 'B'][output['flip_output']]]
        C2 = example['C' + ['B', 'A'][output['flip_output']]]
        print('Judge\'s "R":')
        print(strip_turn(R))

        print('\nJudge\'s "C1":')
        for turn in C1:
            print(strip_turn(turn))

        print('\nJudge\'s "C2":')
        for turn in C2:
            print(strip_turn(turn))

        print('')
    else:
        assert(False)

    #print gt
    print('gt = %s'%(gt))
    print('')

    #print responses
    for kk in sorted(results_dict.keys(), key = lambda kk: kk_val(kk)):
        output = results_dict[kk]['outputs'][k]
        if challenge_type != 'pairwise':
            output = output[i]

        print(interpret_kk(kk) + ' (%s)'%(['incorrect', 'correct'][output['response']['pred'] == gt]) + ':')
        print(output['response'])
        print('')


def is_valid_example(challenge_type, k, results_dict):
    if challenge_type != 'pairwise':
        k, i = k

    flip = None
    for kk in sorted(results_dict.keys()):
        results = results_dict[kk]
        if k not in results['outputs']:
            return False

        output = results['outputs'][k]
        if challenge_type != 'pairwise':
            output = output[i]
            my_flip = output['flip_output']
        else:
            my_flip = (output['flip_input'], output['flip_output'])

        if flip is None:
            flip = my_flip
        else:
            if my_flip != flip:
                return False

    return True


def load_all_results(challenge_type, alm_names, para_cue_types, last_onlys):
    results_dict = {}
    for alm_name in alm_names:
        for para_cue_type in para_cue_types:
            for last_only in last_onlys:
                kk = (alm_name, para_cue_type, last_only)
                results_filename = get_results_filename(challenge_type, para_cue_type, last_only, alm_name)
                with open(results_filename, 'r') as f:
                    results = json.load(f)

                results_dict[kk] = results

    return results_dict


def pick_conversation_example_results(challenge_type, alm_names, para_cue_types, last_onlys, num_examples, random_seed):
    alm_names = alm_names.split(',')
    para_cue_types = para_cue_types.split(',')
    last_onlys = [int(last_only) for last_only in last_onlys.split(',')]
    num_examples = int(num_examples)
    random_seed = int(random_seed)

    random.seed(random_seed)

    examples = load_examples()
    results_dict = load_all_results(challenge_type, alm_names, para_cue_types, last_onlys)
    already = []
    for t in tqdm(range(num_examples)):
        while True:
            k = random.choice(sorted(examples.keys()))
            if challenge_type != 'pairwise':
                i = random.choice([0, 1])
                k = (k, i)

            if is_valid_example(challenge_type, k, results_dict):
                break

        print_example_result(challenge_type, k, examples, results_dict)
        print('\n\n')


def usage():
    print('Usage: python pick_conversation_example_results.py <challenge_type> <alm_names> <para_cue_types> <last_onlys> <num_examples> <random_seed>')


if __name__ == '__main__':
    pick_conversation_example_results(*(sys.argv[1:]))
