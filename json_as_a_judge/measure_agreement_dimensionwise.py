import os
import sys
import json
import math
from tqdm import tqdm
from measure_audiojudge_agreement_dimensionwise import measure_audiojudge_agreement_dimensionwise


def measure_agreement_dimensionwise(results_dict_filename, annotation_dict_filename):
    measure_audiojudge_agreement_dimensionwise(results_dict_filename, annotation_dict_filename, is_audiojudge=False)


def usage():
    print('Usage: python measure_agreement_dimensionwise.py <results_dict_filename> <annotation_dict_filename>')


if __name__ == '__main__':
    measure_agreement_dimensionwise(*(sys.argv[1:]))
