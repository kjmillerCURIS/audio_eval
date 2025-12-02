import os
import sys
import json
from tqdm import tqdm
sys.path.append('.')
from run_alm_on_conversations import get_results_filename, compute_correct_and_total


def compute_accuracy(results):
    correct, total = compute_correct_and_total(results)
    return {m : 100.0 * correct[m] / total[m] for m in ['all', 'typical_order', 'atypical_order']}, total['all']


def print_one_table(alm_name, challenge_type, show_bias):
    N_list = []
    rows = [',no paralinguistic cue,"soft" paralinguistic cue,"hard" paralinguistic cue']
    for last_only in [0, 1]:
        row = [['full convo', 'last turn only'][last_only]]
        for para_cue_type in ['no_para_cue', 'soft_para_cue', 'hard_para_cue']:
            results_filename = get_results_filename(challenge_type, para_cue_type, last_only, alm_name)
            with open(results_filename, 'r') as f:
                results = json.load(f)

            accuracy, N = compute_accuracy(results)
            N_list.append(N)
            if show_bias:
                row.append('%.1f / %.1f'%(accuracy['typical_order'], accuracy['atypical_order']))
            else:
                row.append('%.1f'%(accuracy['all']))

        rows.append(','.join(row))

    print('alm_name="%s"'%(alm_name))
    print('challenge_type="%s"'%(challenge_type))
    for row in rows:
        print(row)

    print('N_list=%s'%(str(N_list)))


def print_conversation_results_tables(show_bias):
    show_bias = int(show_bias)

    for alm_name in ['gemini-2.5-flash', 'gpt4o']:
        for challenge_type in ['pairwise', 'pointwiseCR', 'pointwiseRC']:
            print_one_table(alm_name, challenge_type, show_bias)
            print('\n')


def usage():
    print('Usage: python print_conversation_results_tables.py <show_bias>')


if __name__ == '__main__':
    print_conversation_results_tables(*(sys.argv[1:]))
