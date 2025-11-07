import os
import sys
import json
import math
from tqdm import tqdm


def get_bucketID(example_info):
    if 'language' not in example_info:
        if 'crosstalk' in example_info['input_path']:
            language = 'Chinese'
        else:
            language = 'English'
    else:
        language = example_info['language']

    task = example_info['task'].replace('detections', 'detection')
    task = task.replace(' ', '_')
    return task + '-' + language


def convert_label(label):
    return {'1' : 0, '2' : 1, 'tie' : 0.5}[label]


def compute_credit(output, is_audiojudge=False):
    if is_audiojudge:
        label12, label21 = output['response12']['response']['label'], output['response21']['response']['label']
        assert(label12 in ['1', '2', 'tie'])
        assert(label21 in ['1', '2', 'tie'])
        pred_label = 0.5 * convert_label(label12) + 0.5 * (1 - convert_label(label21))
        gt = output['example_info']['chosen_model']
        assert(gt in ['A', 'B'])
        gt_str = gt
        gt = {'A' : 0, 'B' : 1}[gt]
        credit = 1 - math.fabs(pred_label - gt)
    else:
        pred_label = output['pred']['label']
        assert(pred_label in ['1', '2'])
        pred_label = int(pred_label)
        gt = output['example_info']['chosen_model']
        assert(gt in ['A', 'B'])
        credit = int(['A', 'B'][pred_label - 1] == gt)

    return credit


def measure_agreement_one(bucket, is_audiojudge=False):
    correct, total = 0, 0
    for k in sorted(bucket.keys()):
        output = bucket[k]
        credit = compute_credit(output, is_audiojudge=is_audiojudge)
        correct += credit
        total += 1

    agreement = 100.0 * correct / total
    return (agreement, total)


def stratify_results_one(results_dict_filename):
    is_audiojudge = ('audiojudge' in os.path.splitext(os.path.basename(results_dict_filename))[0].split('_'))
    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

    buckets = {}
    for k in tqdm(sorted(results_dict['outputs'].keys(), key=int)):
        output = results_dict['outputs'][k]
        bucketID = get_bucketID(output['example_info'])
        if bucketID not in buckets:
            buckets[bucketID] = {}

        buckets[bucketID][k] = output

    agreements = {}
    for bucketID in tqdm(sorted(buckets.keys())):
        agreements[bucketID] = measure_agreement_one(buckets[bucketID], is_audiojudge=is_audiojudge)

    return agreements


def stratify_results():
    agreements_dict = {}
    for method in ['AudioJudge+ICL', 'v0_no_aux', 'v1b_no_aux', 'v0_LLM_judge']:
        for rep in [0, 1]:
            if method == 'AudioJudge+ICL':
                if rep > 0:
                    continue

                results_dict_filename = 'json_as_a_judge/s2sarena_experiments/audiojudge_results/audiojudge_results_gpt-4o-audio-preview.json'
            else:
                results_dict_filename = 'json_as_a_judge/s2sarena_experiments/results/results-%s-rep%d.json'%(method, rep)

            agreements_dict[(method, rep)] = stratify_results_one(results_dict_filename)

    for bucketID in sorted(agreements_dict[('v1b_no_aux', 0)].keys()):
        print(bucketID + ':')
        for method in ['AudioJudge+ICL', 'v0_no_aux', 'v1b_no_aux', 'v0_LLM_judge']:
            if method == 'AudioJudge+ICL':
                a, N = agreements_dict[(method, 0)][bucketID]
                print('* %s (N = %d) %.1f%%'%(method, N, a))
            else:
                [a_0, a_1] = [agreements_dict[(method, rep)][bucketID][0] for rep in [0, 1]]
                N_0, N_1 = [agreements_dict[(method, rep)][bucketID][1] for rep in [0, 1]]
                assert(N_0 == N_1)
                a_min, a_max = min(a_0, a_1), max(a_0, a_1)
                print('* %s (N = %d): %.1f%% (%.1f%%, %.1f%%)'%(method, N_0, (a_min + a_max) / 2, a_min, a_max))

        print('')


def usage():
    print('Usage: python stratify_results.py')


if __name__ == '__main__':
    stratify_results()
