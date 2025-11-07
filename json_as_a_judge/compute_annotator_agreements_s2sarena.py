import os
import sys
import math
import pandas as pd
from tqdm import tqdm
from acc_utils import compute_acc


#orig $\leftrightarrow$ blind & - & - & - \\
#orig $\leftrightarrow$ HCoT & - & - & - \\
#blind $\leftrightarrow$ HCoT & - & - & - \\
THREEWAY_MAP = {'1' : '1', '2' : '2', 'both_good' : 'tie', 'both_bad' : 'tie'}
ROW_ORDER = ['blind-vs-orig', 'hcot-vs-orig', 'hcot-vs-blind']
ROWNAME_MAP = {'blind-vs-orig' : 'orig $\\leftrightarrow$ blind', 'hcot-vs-orig' : 'orig $\\leftrightarrow$ HCoT', 'hcot-vs-blind' : 'blind $\\leftrightarrow$ HCoT'}


def compute_annotator_agreements_s2sarena():
    csv_filename = 'json_as_a_judge/s2sarena_experiments/s2sarena_spreadsheets_full_english/s2sarena_LLM_judge_full_english.csv'
    df = pd.read_csv(csv_filename)
    judge_data = {}
    credits = {}
    for metric in ['2-way', '3-way', '4-way']:
        credits[metric] = {'hcot-vs-orig' : [], 'hcot-vs-blind' : [], 'blind-vs-orig' : []}

    for k, row in tqdm(df.iterrows()):
        for kk in ['hcot-vs-orig', 'hcot-vs-blind', 'blind-vs-orig']:
            for metric in ['2-way', '3-way', '4-way']:

                #IMPORTANT: Need to have fresh gts because we modify them!!!
                gts = [str(row['gt_overall_' + z]) for z in kk.split('-vs-')]
                assert(len(gts) == 2)
                assert(all([gt in ['1', '2', 'both_good', 'both_bad'] for gt in gts]))

                if metric in ['3-way', '4-way'] and 'orig' in kk.split('-vs-'): #comparisons with orig can only be 2-way!
                    continue

                if metric == '2-way' and not all([gt in ['1', '2'] for gt in gts]):
                    continue

                if metric == '3-way':
                    gts = [THREEWAY_MAP[gt] for gt in gts]

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
    compute_annotator_agreements_s2sarena()
