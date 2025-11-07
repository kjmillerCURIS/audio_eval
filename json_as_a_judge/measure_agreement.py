import os
import sys
import json
from tqdm import tqdm
from measure_audiojudge_agreement_dimensionwise import CRITERIA, CRITERIA_MAP
from acc_utils import compute_acc


#def measure_agreement(results_dict_filename, annotation_dict_filename=None):
def measure_agreement(results_dict_filename):
    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

#    if annotation_dict_filename is not None:
#        with open(annotation_dict_filename, 'r') as f:
#            annotation_dict = json.load(f)

    credits = []
#    num_ties = 0
    for k in tqdm(sorted(results_dict['outputs'].keys(), key=int)):
        output = results_dict['outputs'][k]
#        if annotation_dict_filename is not None:
#            if k not in annotation_dict:
#                continue
#
#            if not all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] != '' for a in CRITERIA]):
#                assert(all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] == '' for a in CRITERIA]))
#                continue

        pred_label = output['pred']['label']
        if pred_label not in ['1', '2', 'tie']:
            print('unexpected pred_label "%s", skipping'%(pred_label))
            continue

#        if pred_label == 'tie':
#            num_ties += 1

        gt = output['example_info']['chosen_model']
        assert(gt in ['A', 'B'])
        if pred_label == 'tie':
            credit = 0.5
        else:
            pred_label = int(pred_label)
            credit = int(['A', 'B'][pred_label - 1] == gt)

        credits.append(credit)

    acc = compute_acc(credits)
    print('%s (N = %d)'%(acc, len(credits)))


def usage():
#    print('Usage: python measure_agreement.py <results_dict_filename> [<annotation_dict_filename>]')
    print('Usage: python measure_agreement.py <results_dict_filename>')


if __name__ == '__main__':
    measure_agreement(*(sys.argv[1:]))
