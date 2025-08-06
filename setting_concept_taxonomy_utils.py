import os
import sys
import copy
import glob
import json
from pprint import pprint
from tqdm import tqdm
from llm_utils import run_llm


TOPLEVEL_QUERY_TEMPLATE = 'You are an expert at building taxonomies. Please make a taxonomy of information that a %s might request while in the %s setting, or things that might happen to or for the %s while in that setting. Write it first in plaintext and then as a JSON object (spaces in keys are allowed).'
EXPANSION_QUERY_TEMPLATE = 'You are an expert at building taxonomies. Please make a taxonomy within the concept of "%s" which is itself a concept within "%s" in the %s setting. Write it first in plaintext and then as a JSON object (spaces in keys are allowed).'
EXPANSION_QUERY_TEMPLATE_SHALLOWCASE = 'You are an expert at building taxonomies. Please make a taxonomy within the concept of "%s" in the %s setting. Write it first in plaintext and then as a JSON object (spaces in keys are allowed).'


#includes both reformatting and destemming
#this probably modifies my_tree in-place
def sanitize_tree(my_tree, can_destem=True):
    assert(isinstance(my_tree, dict))
    if len(my_tree) == 1:
        if can_destem:
            return sanitize_tree(my_tree[sorted(my_tree.keys())[0]], can_destem=can_destem)
    else:
        can_destem = False

    for k in sorted(my_tree.keys()):
        if my_tree[k] is None or my_tree[k] in [[], {}, True, False, 0, 1] or isinstance(my_tree[k], str):
            my_tree[k] = '' #denotes that we're at a leaf node with value k
        elif isinstance(my_tree[k], list):
            if len(my_tree[k]) == 1 and isinstance(my_tree[k][0], dict):
                my_tree[k] = sanitize_tree(my_tree[k][0], can_destem=can_destem)
            else:
                #print(my_tree[k])
                assert(all([isinstance(s, str) for s in my_tree[k]]))
                my_tree[k] = {s : '' for s in my_tree[k]}
        else:
            #print(my_tree[k])
            assert(isinstance(my_tree[k], dict))
            my_tree[k] = sanitize_tree(my_tree[k], can_destem=can_destem)

    return my_tree


#return leaf_paths, inner_paths
#if include_inner is False then inner_paths will be empty
def find_paths(my_tree, include_inner=True, cur_path=[]):
    assert(isinstance(my_tree, dict))
    leaf_paths, inner_paths = [], []
    for k in sorted(my_tree.keys()):
        my_path = cur_path + [k]
        if my_tree[k] == '': #path ending in k is a leaf path
            leaf_paths.append(copy.deepcopy(my_path))
        else: #path ending in k is an inner path
            if include_inner:
                inner_paths.append(copy.deepcopy(my_path))

            more_leaf_paths, more_inner_paths = find_paths(my_tree[k], include_inner=include_inner, cur_path=copy.deepcopy(my_path))
            leaf_paths.extend(more_leaf_paths)
            inner_paths.extend(more_inner_paths)

    return leaf_paths, inner_paths


#returns (not saves) the taxonomy object
def build_setting_concept_taxonomy(setting_name, user_name):
    toplevel_query = TOPLEVEL_QUERY_TEMPLATE % (user_name, setting_name, user_name)
    my_tree = run_llm(toplevel_query, is_json=True)
    print('here is the toplevel taxonomy:')
    pprint(my_tree)
    my_tree = sanitize_tree(my_tree)
    toplevel_paths, _ = find_paths(my_tree, include_inner=False)
    print('toplevel taxonomy has %d leaf nodes'%(len(toplevel_paths)))
    print('expanding...')
    for p in tqdm(toplevel_paths):
        holder = my_tree
        for k in p[:-1]:
            holder = holder[k]

        if len(p) == 1:
            expansion_query = EXPANSION_QUERY_TEMPLATE_SHALLOWCASE % (p[-1], setting_name)
        else:
            expansion_query = EXPANSION_QUERY_TEMPLATE % (p[-1], ' >> '.join(p[:-1]), setting_name)

        my_subtree = run_llm(expansion_query, is_json=True)
        my_subtree = sanitize_tree(my_subtree)
        holder[p[-1]] = my_subtree

    leaf_paths, inner_paths = find_paths(my_tree, include_inner=True)
    print('full taxonomy has %d leaf nodes and %d inner nodes'%(len(leaf_paths), len(inner_paths)))
    return {'tree' : my_tree, 'leaf_paths' : leaf_paths, 'inner_paths' : inner_paths, 'setting_name' : setting_name, 'user_name' : user_name}


#this part is like a UI
def obtain_setting_concept_taxonomy_phase():
    setting_concept_taxonomy_filenames = sorted(glob.glob('setting_concept_taxonomies/setting_*.json'))
    setting_concept_taxonomies = {}
    print('loading existing setting concept taxonomies...')
    for filename in tqdm(setting_concept_taxonomy_filenames):
        with open(filename, 'r') as f:
            taxonomy = json.load(f)

        index = int(os.path.splitext(os.path.basename(filename))[0].split('_')[-1])
        assert(index not in setting_concept_taxonomies)
        setting_concept_taxonomies[index] = taxonomy

    while True:
        print('Here are the available taxonomies:')
        for index in sorted(setting_concept_taxonomies.keys()):
            print('%d ==> setting_name="%s", user_name="%s"'%(index, setting_concept_taxonomies[index]['setting_name'], setting_concept_taxonomies[index]['user_name']))

        while True:
            my_input = input('Please enter an index of an existing taxonomy, or "+" to build a new one:')
            if my_input == '+' or (my_input.isdigit() and int(my_input) in setting_concept_taxonomies):
                break

        if my_input == '+':
            new_index = max([index for index in sorted(setting_concept_taxonomies.keys())]) + 1 if len(setting_concept_taxonomies) > 0 else 0
            setting_name = input('please enter setting name (e.g. hospital, bank):')
            user_name = input('please enter user name (e.g. patient, customer):')
            taxonomy = build_setting_concept_taxonomy(setting_name, user_name)
            setting_concept_taxonomies[new_index] = taxonomy
            with open(os.path.join('setting_concept_taxonomies', 'setting_%d.json'%(new_index)), 'w') as f:
                json.dump(taxonomy, f, indent=4, sort_keys=True)
        else:
            setting_index = int(my_input)
            return setting_concept_taxonomies[setting_index], setting_index


if __name__ == '__main__':
    setting_name = input('please enter a setting name ')
    user_name = input('please enter a user name ')
    taxonomy = build_setting_concept_taxonomy(setting_name, user_name)
    import pdb
    pdb.set_trace()
