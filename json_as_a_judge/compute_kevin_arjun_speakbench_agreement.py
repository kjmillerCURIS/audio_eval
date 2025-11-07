import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


CRITERIA = ['content', 'voice_quality', 'instruction_following_audio', 'overall']
COLUMN_MAP = {'content' : 'content', 'voice_quality' : 'vq', 'instruction_following_audio' : 'if', 'overall' : 'overall'}
CRITERIA_MAP = {'content' : 'content', 'voice_quality' : 'voice quality', 'instruction_following_audio' : 'paralinguistic EQ/IF', 'overall' : 'overall (looking at dimensions)'}
RATING_MAP = {'1' : '1', '2' : '2', '+' : 'both_good', '-' : 'both_bad'}


def compute_kevin_arjun_speakbench_agreement():
    with open('json_as_a_judge/speakbench_data/speakbench_annotations_kevin.json', 'r') as f:
        annotation_dict = json.load(f)

    for criteria in CRITERIA:
        correct = 0
        total = 0
        singapore_counter = 0
        for k in tqdm(sorted(annotation_dict.keys(), key=int)):
            if k in ['131']:
                continue

            if annotation_dict[k]['ratings']['content'] == '':
                continue

            words = annotation_dict[k]['example_info']['instruction_text'].lower().split()
            if 'singapore' in words or 'singaporean' in words or 'singlish' in words:
                singapore_counter += 1

            kevin_label = RATING_MAP[annotation_dict[k]['ratings'][CRITERIA_MAP[criteria]]]
            arjun_label = annotation_dict[k]['example_info']['label'][criteria]
            assert(arjun_label in ['1', '2', 'both_good', 'both_bad'])
            correct += int(kevin_label == arjun_label)
            total += 1

        print('singapore_counter = %d'%(singapore_counter))
        print('%s exact agreement = %.4f (N = %d)'%(criteria, correct / total, total))

    rows = []
    column_names = ['index', 'instruction_text', 'model_a', 'model_b', 'arjun_content', 'arjun_vq', 'arjun_if', 'arjun_overall', 'kevin_content', 'kevin_vq', 'kevin_if', 'kevin_overall']
    for k in tqdm(sorted(annotation_dict.keys(), key=int)):
        if k in ['131']:
            continue

        if annotation_dict[k]['ratings']['content'] == '':
            continue

        row = {'index' : k, 'instruction_text' : annotation_dict[k]['example_info']['instruction_text'], 'model_a' : annotation_dict[k]['example_info']['model_a'], 'model_b' : annotation_dict[k]['example_info']['model_b']}
        for criteria in CRITERIA:
            row['arjun_' + COLUMN_MAP[criteria]] = annotation_dict[k]['example_info']['label'][criteria]
            row['kevin_' + COLUMN_MAP[criteria]] = RATING_MAP[annotation_dict[k]['ratings'][CRITERIA_MAP[criteria]]]

        rows.append(row)

    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv('json_as_a_judge/speakbench_data/speakbench_annotations_kevin.csv')


if __name__ == '__main__':
    compute_kevin_arjun_speakbench_agreement()
