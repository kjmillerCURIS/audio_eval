import os
import sys
import json
import numpy as np
from tqdm import tqdm


QUALITY_PREFIX = 'json_as_a_judge/s2sarena_experiments/results/results-v0_no_aux'


def get_mean_audio_quality(my_audio_quality_scores):
    return np.mean([float(my_audio_quality_scores[k].split()[0]) for k in sorted(my_audio_quality_scores.keys())])


def break_tie(quality):
    scoreA = get_mean_audio_quality(quality['outputA_json']['agent_audio_quality'])
    scoreB = get_mean_audio_quality(quality['outputB_json']['agent_audio_quality'])
    assert(scoreA != scoreB)
    if scoreA > scoreB:
        return '1'
    else:
        return '2'


def measure_agreement_break_ties_with_quality(results_dict_filename, first_n=None):
    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

    rep_part = os.path.splitext(os.path.basename(results_dict_filename))[0].split('-')[-1]
    assert(rep_part[:3] == 'rep')
    quality_dict_filename = QUALITY_PREFIX + '-' + rep_part + '.json'
    with open(quality_dict_filename, 'r') as f:
        quality_dict = json.load(f)

    if first_n is None:
        first_n = len(results_dict['outputs'])
    else:
        first_n = int(first_n)

    total = 0
    correct = 0
    num_ties_broken = 0
    for k in tqdm(sorted(results_dict['outputs'].keys(), key=int)[:first_n]):
        output = results_dict['outputs'][k]
        pred_label = output['pred']['label']
        if pred_label not in ['1', '2', 'tie']:
            print('unexpected pred_label "%s", skipping'%(pred_label))
            continue

        if pred_label == 'tie':
            pred_label = break_tie(quality_dict['outputs'][k])
            num_ties_broken += 1

        assert(pred_label in ['1', '2'])
        gt = output['example_info']['chosen_model']
        assert(gt in ['A', 'B'])
        pred_label = int(pred_label)
        credit = int(['A', 'B'][pred_label - 1] == gt)

        correct += credit
        total += 1

    agreement = 100.0 * correct / total
    print('"%s": agreement = %.1f%% (%.1f / %d) (%d ties broken)'%(results_dict_filename, agreement, correct, total, num_ties_broken))

    return agreement


def usage():
    print('Usage: python measure_agreement_break_ties_with_quality.py <results_dict_filename> [<first_n>]')


if __name__ == '__main__':
    measure_agreement_break_ties_with_quality(*(sys.argv[1:]))
