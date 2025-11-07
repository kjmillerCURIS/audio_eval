import os
import sys
import json


EXAMPLE_INFO_DICT_FILENAME = 'json_as_a_judge/s2sarena_experiments/human_evaluate_result.json'


def load_example_info_dict():
    with open(EXAMPLE_INFO_DICT_FILENAME, 'r') as f:
        example_info_list = json.load(f)

    example_info_dict = {str(i) : example_info for i, example_info in enumerate(example_info_list)}
    return example_info_dict
