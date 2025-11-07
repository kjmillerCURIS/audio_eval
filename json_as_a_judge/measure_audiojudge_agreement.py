import os
import sys
import json
import math
from tqdm import tqdm
from measure_audiojudge_agreement_dimensionwise import CRITERIA, CRITERIA_MAP
from acc_utils import compute_acc


def convert_label(label):
    return {'1' : 0, '2' : 1, 'tie' : 0.5}[label]


#def measure_audiojudge_agreement(results_dict_filename, annotation_dict_filename=None):
def measure_audiojudge_agreement(results_dict_filename):
    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

#    if annotation_dict_filename is not None:
#        with open(annotation_dict_filename, 'r') as f:
#            annotation_dict = json.load(f)

    credits = []
    for k in tqdm(sorted(results_dict['outputs'].keys(), key=int)):
        output = results_dict['outputs'][k]
#        if annotation_dict_filename is not None:
#            if k not in annotation_dict:
#                continue
#
#            if not all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] != '' for a in CRITERIA]):
#                assert(all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] == '' for a in CRITERIA]))
#                continue

        label12, label21 = output['response12']['response']['label'], output['response21']['response']['label']
        assert(label12 in ['1', '2', 'tie'])
        assert(label21 in ['1', '2', 'tie'])
        pred_labels = (convert_label(label12), 1 - convert_label(label21))
        gt = output['example_info']['chosen_model']
        assert(gt in ['A', 'B'])
        gt_str = gt
        gt = {'A' : 0, 'B' : 1}[gt]
        credit = sum([1 - math.fabs(pred_label - gt) for pred_label in pred_labels]) / len(pred_labels)
        credits.append(credit)

    acc = compute_acc(credits)
    print('%s (N = %d)'%(acc, len(credits)))


def usage():
#    print('Usage: python measure_audiojudge_agreement.py <results_dict_filename> [<annotation_dict_filename>]')
    print('Usage: python measure_audiojudge_agreement.py <results_dict_filename>')


if __name__ == '__main__':
    measure_audiojudge_agreement(*(sys.argv[1:]))
