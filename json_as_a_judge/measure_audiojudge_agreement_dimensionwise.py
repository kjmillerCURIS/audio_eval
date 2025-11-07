import os
import sys
import json
import math
import numpy as np
from tqdm import tqdm
from acc_utils import compute_acc


#CRITERIA = ['content', 'voice_quality', 'instruction_following_audio', 'overall (looking at dimensions)']
CRITERIA = ['content', 'voice_quality', 'instruction_following_audio']
CRITERIA_MAP = {'content' : 'content', 'voice_quality' : 'voice quality', 'instruction_following_audio' : 'paralinguistic EQ/IF', 'overall (looking at dimensions)' : 'overall (looking at dimensions)'}
RATING_MAP = {'1' : '1', '2' : '2', '+' : 'both_good', '-' : 'both_bad'}


def measure_audiojudge_agreement_dimensionwise(results_dict_filename, annotation_dict_filename, is_audiojudge=True):
    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

    with open(annotation_dict_filename, 'r') as f:
        annotation_dict = json.load(f)

    credits = {a : [] for a in CRITERIA}
    content_confusion_matrix = np.zeros((4,4))
    for k in tqdm(sorted(results_dict['outputs'].keys(), key=int)):
        output = results_dict['outputs'][k]
        if k not in annotation_dict:
            continue

        if not all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] != '' for a in CRITERIA]):
            assert(all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] == '' for a in CRITERIA]))
            continue

        for criteria in CRITERIA:
            credit = 0
            preds = []
            if is_audiojudge:
                for swap in ['12', '21']:
                    if criteria == 'overall (looking at dimensions)':
                        pred = output['fusion_part']['response' + swap]['pred']['overall_label']
                    else:
                        pred = output['response' + swap]['response'][criteria]

                    assert(pred in ['1', '2', 'both_good', 'both_bad'])
                    if swap == '21' and pred in ['1', '2']:
                        pred = {'1' : '2', '2' : '1'}[pred]

                    preds.append(pred)
            else:
                if criteria == 'overall (looking at dimensions)':
                    pred = output['fusion_part']['pred']['overall_label']
                else:
                    pred = output['pred'][criteria]
                assert(pred in ['1', '2', 'both_good', 'both_bad'])
                preds.append(pred)

            gt = annotation_dict[k]['ratings'][CRITERIA_MAP[criteria]]
            gt = RATING_MAP[gt]
            for pred in preds:
                credit += int(pred == gt)

            credit /= len(preds)
            credits[criteria].append(credit)

            if criteria == 'content':
                i = ['1', '2', 'both_good', 'both_bad'].index(gt)
                for pred in preds:
                    j = ['1', '2', 'both_good', 'both_bad'].index(pred)
                    content_confusion_matrix[i,j] += 1 / len(preds)

    for criteria in CRITERIA:
        acc = compute_acc(credits[criteria])
        print('%s: %s (N = %d)'%(criteria, acc, len(credits[criteria])))


def usage():
    print('Usage: python measure_audiojudge_agreement_dimensionwise.py <results_dict_filename> <annotation_dict_filename>')


if __name__ == '__main__':
    measure_audiojudge_agreement_dimensionwise(*(sys.argv[1:]))
