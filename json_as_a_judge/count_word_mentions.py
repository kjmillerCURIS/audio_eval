import os
import sys
import json
from tqdm import tqdm


def count_word_mentions(results_dict_filename, word):
    with open(results_dict_filename, 'r') as f:
        results_dict = json.load(f)

    mentioned = 0
    for k in tqdm(sorted(results_dict['outputs'].keys(), key=int)):
        reasoning = results_dict['outputs'][k]['pred']['reasoning']
        if word.lower() in reasoning.lower():
            mentioned += 1

    N = len(results_dict['outputs'])
    print('%.1f%% of examples mentioned "%s" (%d / %d)'%(100.0 * mentioned / N, word, mentioned, N))


def usage():
    print('Usage: python count_word_mentions.py <results_dict_filename> <word>')


if __name__ == '__main__':
    count_word_mentions(*(sys.argv[1:]))
