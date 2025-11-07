import os
import sys
import json
import shutil
from tqdm import tqdm
from s2sarena_utils import load_example_info_dict


#  {
#    "category": "Social Companionship",
#    "id": "irony_irony1",
#    "input_path": "input/irony/irony1.wav",
#    "language": "Chinese",
#    "level": "L1",
#    "output_path_4o": "output/ChatGPT-4o/irony/irony1/irony1.wav",
#    "output_path_4o_cascade": "output/cascade/irony/irony1.wav",
#    "output_path_4o_llama_omni": "output/LLaMA_omni/irony/irony1.wav",
#    "output_path_funaudio": "output/FunAudioLLM/irony/audio_0.wav",
#    "output_path_miniomni": "output/Mini-Omni/irony/00.wav",
#    "output_path_speechgpt": "output/SpeechGPT/irony/irony1.wav",
#    "task": "Sarcasm detection",
#    "task_description": "Can the model detect sarcasm in phrases like “You're amazing!”?",
#    "text": "昨天小明用这个语气对我说“你可真聪明！”，他是什么意思？",
#    "name": "expert_14",
#    "chosen_model": "B",
#    "model_a": "output_path_miniomni",
#    "model_b": "output_path_4o_cascade",
#    "result": {
#      "output_path_miniomni": 0,
#      "output_path_4o_cascade": 1
#    }
#  },
VA_LIST = ['4o', '4o_cascade', '4o_llama_omni', 'funaudio', 'miniomni', 'speechgpt']
BAD_KEYS = ['414', '416', '417', '419']


def get_bucket_id(example_info):
    language = 'NA'
    if 'language' in example_info:
        language = example_info['language']

    task = example_info['task'].replace(' ', '_')
    return task + '-' + language


def get_structure(example_info_dict):
    input2pairs = {}
    structure = {}
    for k in tqdm(sorted(example_info_dict, key=int)):
        if k in BAD_KEYS or 'input/noise' in example_info_dict[k]['input_path'] or example_info_dict[k]['input_path'] == 'input/rhythm/input_audio_3.mp3':
            continue

        my_input = example_info_dict[k]['input_path']
        if my_input not in structure:
            bucket_id = get_bucket_id(example_info_dict[k])
            structure[my_input] = {'input' : my_input, 'outputs' : {}, 'k' : k, 'bucket_id' : bucket_id}

        for va in VA_LIST:
            if 'output_path_' + va not in example_info_dict[k]:
                continue

            my_output = example_info_dict[k]['output_path_' + va].lstrip('/').replace('tongue twisters', 'tongue_twister')
            if va in structure[my_input]['outputs']:
                if structure[my_input]['outputs'][va] != my_output:
                    print(k)
                    print(structure[my_input]['outputs'][va])
                    print(my_output)
                    assert(False)
            else:
                structure[my_input]['outputs'][va] = my_output

    return structure


def do_copying(structure):
    parent_dir = 'json_as_a_judge/s2sarena_experiments/organized_audio_files'
    for my_input in tqdm(sorted(structure.keys())):
        dst_dir = os.path.join(parent_dir, structure[my_input]['bucket_id'])
        os.makedirs(dst_dir, exist_ok=True)
        prefix = structure[my_input]['bucket_id'] + '-' + structure[my_input]['k']
        input_filename = os.path.join(dst_dir, prefix + '-input' + os.path.splitext(structure[my_input]['input'])[-1])
        shutil.copy(os.path.join('json_as_a_judge/s2sarena_experiments/audio_files', structure[my_input]['input']), input_filename)
        for va in VA_LIST:
            output_filename = os.path.join(dst_dir, prefix + '-output-' + va + os.path.splitext(structure[my_input]['outputs'][va])[-1])
            shutil.copy(os.path.join('json_as_a_judge/s2sarena_experiments/audio_files', structure[my_input]['outputs'][va]), output_filename)


def organize_s2sarena_audio_files():
    example_info_dict = load_example_info_dict()
    structure = get_structure(example_info_dict)
    do_copying(structure)


if __name__ == '__main__':
    organize_s2sarena_audio_files()
