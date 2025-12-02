import os
import sys
import json
from tqdm import tqdm
import pdb


def print_params(results):
    params = {k : results[k] for k in sorted(results.keys()) if k != 'outputs'}
    print(params)


def conversation_results_explorer(results_filename, correct_or_incorrect):
    assert(correct_or_incorrect in ['correct', 'incorrect'])

    with open(results_filename, 'r') as f:
        results = json.load(f)

    print_params(results)
    is_pairwise = (results['challenge_type'] == 'pairwise')
    my_keys = []
    for k in tqdm(sorted(results['outputs'].keys())):
        output = results['outputs'][k]
        if is_pairwise:
            assert(output['credit'] in [0, 1])
            if output['credit'] == {'correct' : 1, 'incorrect' : 0}[correct_or_incorrect]:
                my_keys.append(k)
        else:
            for i, o in enumerate(output):
                assert(o['credit'] in [0, 1])
                if o['credit'] == {'correct' : 1, 'incorrect' : 0}[correct_or_incorrect]:
                    my_keys.append((k, i))

    print(my_keys)

    while True:
        s = input('meow?:')
        if s == 'i':
            print_params(results)
        elif s == 'k':
            print(my_keys)
        else:
            if is_pairwise:
                if s in my_keys:
                    output = results['outputs'][s]
                    print(output.keys())
                    pdb.set_trace()
            else:
                ss = s.split(',')
                if len(ss) == 2 and ss[1].isdigit() and (ss[0], int(ss[1])) in my_keys:
                    output = results['outputs'][ss[0]][int(ss[1])]
                    print(output.keys())
                    pdb.set_trace()


def usage():
    print('Usage: python conversation_results_explorer.py <results_filename> <correct_or_incorrect>')


if __name__ == '__main__':
    conversation_results_explorer(*(sys.argv[1:]))
