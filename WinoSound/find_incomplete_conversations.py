import os
import sys
import glob
import json
from tqdm import tqdm


def find_incomplete_conversations(parent_dir):
    example_dirs = sorted(glob.glob(os.path.join(parent_dir, '*')))
    for example_dir in tqdm(example_dirs):
        if not os.path.isdir(example_dir):
            continue

        info_filename = os.path.join(example_dir, os.path.basename(example_dir) + '-info.json')
        if not os.path.exists(info_filename):
            print('missing info file for "%s"'%(example_dir))
            continue

        with open(info_filename, 'r') as f:
            example = json.load(f)

        for AB in ['A', 'B']:
            if 'C' + AB not in example or 'R' + AB not in example:
                print('missing field in info file for "%s"'%(example_dir))
                break

            if example['R' + AB]['audio_path'] == 'NA':
                print('missing audio path for "%s"'%(example_dir))
                break

            for turn in example['C' + AB]:
                if turn['audio_path'] == 'NA':
                    print('missing audio path for "%s"'%(example_dir))
                    break


def usage():
    print('Usage: python find_incomplete_conversations.py <parent_dir>')


if __name__ == '__main__':
    find_incomplete_conversations(*(sys.argv[1:]))
