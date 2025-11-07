import os
import sys
import math
import pandas as pd
from tqdm import tqdm
from acc_utils import compute_acc


#orig $\leftrightarrow$ blind & - & - & - \\
#orig $\leftrightarrow$ HCoT & - & - & - \\
#blind $\leftrightarrow$ HCoT & - & - & - \\
THREEWAY_MAP = {'1' : '1', '2' : '2', 'both_good' : 'tie', 'both_bad' : 'tie', 'tie' : 'tie'}
ROW_ORDER = ['blind-vs-orig', 'hcot-vs-orig', 'hcot-vs-blind']
ROWNAME_MAP = {'blind-vs-orig' : 'orig $\\leftrightarrow$ blind', 'hcot-vs-orig' : 'orig $\\leftrightarrow$ HCoT', 'hcot-vs-blind' : 'blind $\\leftrightarrow$ HCoT'}


#return dict of dicts with top-level keys 'hcot', 'orig', 'blind'
#lower-level keys should be in string form
#labels themselves should also be in string form
def load_labels():
    labels = {'hcot' : {}, 'orig' : {}, 'blind' : {}}
    dfA = pd.read_csv('json_as_a_judge/SpeakBench_Dimensionwise_Reasoning/labels_no_hcot.csv')
    for _, row in tqdm(dfA.iterrows()):
        k = str(row['index'])
        assert(all([k not in labels[z] for z in ['orig', 'blind']]))
        orig_label = str(row['original_label'])
        if orig_label not in ['1', '2', 'tie']:
            print(row)
        else:
            labels['orig'][k] = orig_label

        blind_label = str(row['arjun_label'])
        if blind_label not in ['1', '2', 'both_good', 'both_bad']:
            print(row)
        else:
            labels['blind'][k] = blind_label

    dfB = pd.read_csv('json_as_a_judge/SpeakBench_Dimensionwise_Reasoning/json_judge_hcot_fusion.csv')
    for _, row in tqdm(dfB.iterrows()):
        k = str(row['index'])
        assert(all([k not in labels[z] for z in ['hcot']]))
        hcot_label = str(row['gt_overall'])
        if hcot_label not in ['1', '2', 'both_good', 'both_bad']:
            print(row)
        else:
            labels['hcot'][k] = hcot_label

    return labels


def compute_annotator_agreements_speakbench():
    labels = load_labels()
    my_keys = set(labels['hcot'].keys())
    my_keys = my_keys.union(set(labels['orig'].keys()))
    my_keys = my_keys.union(set(labels['blind'].keys()))
    my_keys = sorted(my_keys, key=int)

    credits = {}
    for metric in ['2-way', '3-way', '4-way']:
        credits[metric] = {'hcot-vs-orig' : [], 'hcot-vs-blind' : [], 'blind-vs-orig' : []}

    for k in tqdm(my_keys):
        for kk in ['hcot-vs-orig', 'hcot-vs-blind', 'blind-vs-orig']:
            for metric in ['2-way', '3-way', '4-way']:
                if not all([k in labels[z] for z in kk.split('-vs-')]):
                    continue

                #IMPORTANT: Need to have fresh gts because we modify them!!!
                gts = [labels[z][k] for z in kk.split('-vs-')]
                assert(len(gts) == 2)
                assert(all([gt in ['1', '2', 'both_good', 'both_bad', 'tie'] for gt in gts]))

                if metric == '4-way' and 'orig' in kk.split('-vs-'): #comparisons with orig cannot be 4-way!
                    continue

                if metric == '2-way' and not all([gt in ['1', '2'] for gt in gts]):
                    continue

                if metric == '3-way':
                    gts = [THREEWAY_MAP[gt] for gt in gts]

                if metric == '4-way':
                    assert(all([gt in ['1', '2', 'both_good', 'both_bad'] for gt in gts]))

                credit = int(gts[0] == gts[1])
                credits[metric][kk].append(credit)

    for kk in ROW_ORDER:
        line = ROWNAME_MAP[kk]
        entries = []
        for metric in ['2-way', '3-way', '4-way']:
            if len(credits[metric][kk]) == 0:
                entries.append('-')
                continue

            entries.append(compute_acc(credits[metric][kk], fancy=True))

        line = line + ' & ' + ' & '.join(entries) + ' \\\\'
        print(line)


if __name__ == '__main__':
    compute_annotator_agreements_speakbench()
