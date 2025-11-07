import os
import sys
import copy
import json
import numpy as np
from string import Template
from tqdm import tqdm
sys.path.append('.')
from llm_utils import run_llm


USE_SAVED_PROGRESS = True
BAD_KEYS = []


def make_prompt(json_analysis):
    prompt_template_filename = 'json_as_a_judge/prompts/pairwise_prompt_s2sarena_fusion.txt'
    with open(prompt_template_filename, 'r') as f:
        prompt_template = f.read()

    prompt_template = Template(prompt_template)
    prompt = prompt_template.substitute(json_analysis=json_analysis)
    return prompt


def run_inference(prompt, llm_name):
    pred, full_response = run_llm(prompt, is_json=True, llm_name=llm_name, dimensionwise=2)
    if pred is None:
        return None, None

    return pred, full_response


def s2sarena_fusion_step(results_dict_filename, llm_name, is_audiojudge):
    is_audiojudge = int(is_audiojudge)
    assert(llm_name in os.path.splitext(os.path.basename(results_dict_filename))[0])

    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

    print('total of %d examples'%(len(results_dict['outputs'])))
    if not is_audiojudge:
        assert(llm_name == results_dict['llm_name'])

    if is_audiojudge:
        fusion_dict = {}
    else:
        fusion_dict = {kk : results_dict[kk] for kk in ['params_key', 'params', 'llm_name', 'dimensionwise']}

    fusion_dict['outputs'] = {}
    fusion_dict_filename = os.path.splitext(results_dict_filename)[0] + '-with_fusion.json'
    if USE_SAVED_PROGRESS and os.path.exists(fusion_dict_filename):
        with open(fusion_dict_filename, 'r') as f:
            fusion_dict = json.load(f)

        print('oh goody! found %d examples already computed!'%(len(fusion_dict['outputs'])))

    for t, k in tqdm(enumerate(sorted(results_dict['outputs'].keys(), key=int))):
        if USE_SAVED_PROGRESS and k in fusion_dict['outputs']:
            print('skipping "%s" (because saved!)'%(k))
            continue

        if k in BAD_KEYS:
            print('skipping "%s" (because bad!)'%(k))
            continue

        print('computing "%s"...'%(k))
        output = results_dict['outputs'][k]
        if is_audiojudge:
            fusion_part = {}
            is_error = False
            for swap in ['12', '21']:
                prompt = make_prompt(output['response' + swap]['response'])
                pred = None
                pred, full_response = run_inference(prompt, llm_name)
                if pred is None: #try again...
                    pred, full_response = run_inference(prompt, llm_name)

                if pred is None:
                    is_error = True
                    break

                fusion_part['response' + swap] = {'pred' : pred, 'full_response' : full_response}

            if is_error:
                print('skipping "%s" (because error!)'%(k))
                continue

            output['fusion_part'] = fusion_part
        else:
            prompt = make_prompt(output['pred'])
            pred = None
            pred, full_response = run_inference(prompt, llm_name)
            if pred is None: #try again...
                pred, full_response = run_inference(prompt, llm_name)

            if pred is None:
                print('skipping "%s" (because error!)'%(k))
                continue

            output['fusion_part'] = {'pred' : pred, 'full_response' : full_response}

        fusion_dict['outputs'][k] = output
        if t == 1 or (t > 0 and t % 5 == 0):
            with open(fusion_dict_filename, 'w') as f:
                json.dump(fusion_dict, f)

    with open(fusion_dict_filename, 'w') as f:
        json.dump(fusion_dict, f)


def usage():
    print('Usage: python s2sarena_fusion_step.py <results_dict_filename> <llm_name> <is_audiojudge>')


if __name__ == '__main__':
    s2sarena_fusion_step(*(sys.argv[1:]))
