import os
import sys
import json
import math
import pandas as pd
from tqdm import tqdm
from apply_rules_to_s2sarena_final import majority_vote_fn
from acc_utils import compute_acc


def measure_agreement_one_judge_one_rule(judge_data, key_list, rule_fn):
    credits = []
    for k in tqdm(key_list):
        row = judge_data[k]
        rating_vec = {'content' : row['prediction_content'], 'voice_quality' : row['prediction_vq'], 'instruction_following_audio' : row['prediction_if'], 'LLM_fused' : row['prediction_overall']}
        pred = rule_fn(rating_vec)
        gt = row['gt_overall']
        assert(pred in ['1', '2', 'both_good', 'both_bad'])
        assert(gt in ['1', '2', 'both_good', 'both_bad'])
        credits.append(int(pred == gt))

    return compute_acc(credits)


#dict mapping keys (row indices) to dicts that use column names as keys
#any missing values ==> skip row
def load_spreadsheet(csv_filename):
    df = pd.read_csv(csv_filename)
    judge_data = {}
    required_columns = ['prediction_content', 'prediction_if', 'prediction_vq', 'prediction_overall', 'gt_overall', 'gt_content']
    for k, row in df.iterrows():
        should_skip = False
        for c in required_columns:
            if row[c] not in ['1', '2', 'both_good', 'both_bad']:
                should_skip = True
                break

        if should_skip:
            continue

        judge_data[k] = row

    return judge_data


def apply_rules_to_speakbench_final():
    AudioJudge_csv_filename = 'json_as_a_judge/SpeakBench_HCoT_Results/audio_judge_hcot_fusion.csv'
    LLM_judge_csv_filename = 'json_as_a_judge/SpeakBench_HCoT_Results/llm_judge_hcot_fusion.csv'
    JSON_judge_csv_filename = 'json_as_a_judge/SpeakBench_HCoT_Results/json_judge_hcot_fusion.csv'

    AudioJudge_data = load_spreadsheet(AudioJudge_csv_filename)
    LLM_judge_data = load_spreadsheet(LLM_judge_csv_filename)
    JSON_judge_data = load_spreadsheet(JSON_judge_csv_filename)

    key_list = set(AudioJudge_data.keys())
    key_list = key_list.intersection(set(LLM_judge_data.keys()))
    key_list = key_list.intersection(set(JSON_judge_data.keys()))
    key_list = sorted(key_list)
    print('%d examples'%(len(key_list)))

    lines = []
    for rule_fn in [majority_vote_fn]:
        AudioJudge_agreement = measure_agreement_one_judge_one_rule(AudioJudge_data, key_list, rule_fn)
        LLM_judge_agreement = measure_agreement_one_judge_one_rule(LLM_judge_data, key_list, rule_fn)
        JSON_judge_agreement = measure_agreement_one_judge_one_rule(JSON_judge_data, key_list, rule_fn)
        lines.append('"%s": AudioJudge = %s, LLM-judge = %s, JSON-judge = %s'%(rule_fn.__name__, AudioJudge_agreement, LLM_judge_agreement, JSON_judge_agreement))

    print('\n'.join(lines))


if __name__ == '__main__':
    apply_rules_to_speakbench_final()
