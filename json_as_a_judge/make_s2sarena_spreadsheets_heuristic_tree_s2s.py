import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from apply_rules_to_s2sarena_final import heuristic_tree_s2s_fn


#key, text, model_a, model_b, [swap], prediction_content, prediction_vq, prediction_if, prediction_overall_tree, gt_overall_orig, gt_content, gt_vq, gt_if, gt_overall_hcot, gt_overall_blind, dimensionwise_reasoning
CRITERIA = ['content', 'voice_quality', 'instruction_following_audio', 'overall (looking at dimensions)', 'overall (not looking at dimensions)']
CRITERIA_MAP = {'content' : 'content', 'voice_quality' : 'voice quality', 'instruction_following_audio' : 'paralinguistic EQ/IF', 'overall (looking at dimensions)' : 'overall (looking at dimensions)', 'overall (not looking at dimensions)' : 'overall (not looking at dimensions)'}
COLUMN_CRITERIA_MAP = {'content' : 'content', 'voice_quality' : 'vq', 'instruction_following_audio' : 'if', 'overall (looking at dimensions)' : 'overall_hcot', 'overall (not looking at dimensions)' : 'overall_blind'}
RATING_MAP = {'1' : '1', '2' : '2', '+' : 'both_good', '-' : 'both_bad'}


def make_row(k, results_dict, annotation_dict, is_audiojudge, swap=None):
    row = {}
    output, ratings = results_dict['outputs'][k], annotation_dict[k]['ratings']
    example_info = output['example_info']
    for kk in ['text', 'model_a', 'model_b']:
        row[kk] = example_info[kk]

    row['gt_overall_orig'] = {'A' : '1', 'B' : '2'}[example_info['chosen_model']]
    for a in CRITERIA:
        row['gt_' + COLUMN_CRITERIA_MAP[a]] = RATING_MAP[ratings[CRITERIA_MAP[a]]]

    row['key'] = k
    if is_audiojudge:
        row['swap'] = swap

    rating_vec = {}
    for criteria in CRITERIA:
        if criteria in ['overall (looking at dimensions)', 'overall (not looking at dimensions)']:
            continue

        if is_audiojudge:
            pred = output['response' + swap]['response'][criteria]
            if swap == '21' and pred in ['1', '2']:
                pred = {'1' : '2', '2' : '1'}[pred]
        else:
            pred = output['pred'][criteria]

        assert(pred in ['1', '2', 'both_good', 'both_bad'])
        row['prediction_' + COLUMN_CRITERIA_MAP[criteria]] = pred
        rating_vec[criteria] = pred

    #no need to flip prediction, because the inputs for it have already been flipped
    row['prediction_overall_tree'] = heuristic_tree_s2s_fn(rating_vec)
    if is_audiojudge:
        row['dimensionwise_reasoning'] = output['response' + swap]['response']['reasoning']
    else:
        row['dimensionwise_reasoning'] = output['pred']['reasoning']

    return row


def make_s2sarena_spreadsheets_one_judge(results_dict, annotation_dict, is_audiojudge, csv_filename):
    rows = []
    if is_audiojudge:
        column_names = ['key', 'text', 'model_a', 'model_b', 'swap', 'prediction_content', 'prediction_vq', 'prediction_if', 'prediction_overall_tree', 'gt_overall_orig', 'gt_content', 'gt_vq', 'gt_if', 'gt_overall_hcot', 'gt_overall_blind', 'dimensionwise_reasoning']
    else:
        column_names = ['key', 'text', 'model_a', 'model_b', 'prediction_content', 'prediction_vq', 'prediction_if', 'prediction_overall_tree', 'gt_overall_orig', 'gt_content', 'gt_vq', 'gt_if', 'gt_overall_hcot', 'gt_overall_blind', 'dimensionwise_reasoning']

    credits = []
    for k in tqdm(sorted(annotation_dict.keys(), key=int)):
        assert(k in results_dict['outputs'])
        if any([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] == '' for a in CRITERIA]):
            assert(all([annotation_dict[k]['ratings'][CRITERIA_MAP[a]] == '' for a in CRITERIA]))
            continue

        subcredits = []
        if is_audiojudge:
            for swap in ['12', '21']:
                row = make_row(k, results_dict, annotation_dict, is_audiojudge, swap=swap)
                rows.append(row)
                subcredits.append(int(row['gt_overall_hcot'] == row['prediction_overall_tree']))
        else:
            row = make_row(k, results_dict, annotation_dict, is_audiojudge)
            rows.append(row)
            subcredits.append(int(row['gt_overall_hcot'] == row['prediction_overall_tree']))

        credits.append(np.mean(subcredits))

    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(csv_filename)
    print(np.mean(credits))


def make_s2sarena_spreadsheets_heuristic_tree_s2s():
    data_dir = 'json_as_a_judge/s2sarena_experiments'
    annotation_dict_filename = os.path.join(data_dir, 's2sarena_annotations_kevin.json')
    AudioJudge_results_dict_filename = os.path.join(data_dir, 'audiojudge_results', 'audiojudge_results_gemini-2.5-flash_icl0_dimensionwise1-with_fusion.json')
    LLM_judge_results_dict_filename = os.path.join(data_dir, 'results', 'results-v2_LLM_judge-gemini-2.5-flash-dimensionwise1-rep0-with_fusion.json')
    JSON_judge_results_dict_filename = os.path.join(data_dir, 'results', 'results-v2-gemini-2.5-flash-dimensionwise1-rep0-with_fusion.json')
    with open(annotation_dict_filename, 'r') as f:
        annotation_dict = json.load(f)

    with open(AudioJudge_results_dict_filename, 'r') as f:
        AudioJudge_results_dict = json.load(f)

    with open(LLM_judge_results_dict_filename, 'r') as f:
        LLM_judge_results_dict = json.load(f)

    with open(JSON_judge_results_dict_filename, 'r') as f:
        JSON_judge_results_dict = json.load(f)

    csv_dir = 'json_as_a_judge/s2sarena_experiments/s2sarena_spreadsheets_full_english_heuristic_tree_s2s'
    os.makedirs(csv_dir, exist_ok=True)
    make_s2sarena_spreadsheets_one_judge(AudioJudge_results_dict, annotation_dict, True, os.path.join(csv_dir, 's2sarena_AudioJudge_full_english_heuristic_tree_s2s.csv'))
    make_s2sarena_spreadsheets_one_judge(LLM_judge_results_dict, annotation_dict, False, os.path.join(csv_dir, 's2sarena_LLM_judge_full_english_heuristic_tree_s2s.csv'))
    make_s2sarena_spreadsheets_one_judge(JSON_judge_results_dict, annotation_dict, False, os.path.join(csv_dir, 's2sarena_JSON_judge_full_english_heuristic_tree_s2s.csv'))


if __name__ == '__main__':
    make_s2sarena_spreadsheets_heuristic_tree_s2s()
