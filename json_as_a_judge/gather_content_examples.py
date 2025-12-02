import os
import sys
import json
import random
from tqdm import tqdm


RATING_MAP = {'1' : '1 > 2', '2' : '2 > 1', '+' : 'both-good', '-' : 'both-bad'}



def gather_content_examples(results_dict_filename, annotation_dict_filename, N):
    N = int(N)

    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

    with open(annotation_dict_filename, 'r') as f:
        annotation_dict = json.load(f)

    my_keys = random.sample(sorted(annotation_dict.keys()), N)
    for k in tqdm(my_keys):
        output = results_dict['outputs'][k]
        rating = annotation_dict[k]['ratings']
        if rating['content'] != '1':
            continue

        #'content': '2', 'voice quality': '+', 'paralinguistic EQ/IF': '-', 'overall (looking at dimensions)'
        print('input = "%s", response1 = "%s", response2 = "%s"'%(output['input_json'], output['outputA_json'], output['outputB_json']))
        print('C = "%s", VQ = "%s", EQ = "%s", Overall = "%s"'%(RATING_MAP[rating['content']], RATING_MAP[rating['voice quality']], RATING_MAP[rating['paralinguistic EQ/IF']], RATING_MAP[rating['overall (looking at dimensions)']]))
        print('')


def usage():
    print('Usage: python gather_content_examples.py <results_dict_filename> <annotation_dict_filename> <N>')


if __name__ == '__main__':
    gather_content_examples(*(sys.argv[1:]))
