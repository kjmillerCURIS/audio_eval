import os
import sys
import glob
from tqdm import tqdm


#MODE = 'refinement_logic'
MODE = 'final_final'


def peruse_one_refinement_logic(lines):
    started = False
    ended = False
    for line in lines:
        if ended:
            continue
        if started:
            if '=======FINAL ANSWER=======' in line:
                ended = True
            elif line.strip() == '':
                continue
            else:
                print(line.rstrip('\n'))
        elif '=======AFTER REFINEMENT=======' in line:
            started = True


def peruse_one_final_final(lines):
    after = False
    final_final = False
    last_human_emotions = [None, None]
    cur_conv = 0
    for line in lines:
        if final_final:
            print(line.rstrip('\n'))
            if line.strip() == '===':
                assert(cur_conv == 0)
                cur_conv = 1
            elif 'HUMAN' in line:
                emotion = line.split('HUMAN(')[1].split('):')[0]
                last_human_emotions[cur_conv] = emotion

        elif after:
            if '=======FINAL ANSWER=======' in line:
                final_final = True
        else:
            if '=======AFTER REFINEMENT=======' in line:
                after = True

    assert(None not in last_human_emotions)
    return tuple(sorted(last_human_emotions))


def peruse(log_dir):
    filenames = sorted(glob.glob(os.path.join(log_dir, 'input_aug', '*.txt')))
    if MODE == 'final_final':
        lhep_counter = {}
        meow_counter = {}

    for filename in tqdm(filenames):
        with open(filename, 'r') as f:
            lines = f.read().split('\n')

        print(os.path.basename(filename))
        if MODE == 'refinement_logic':
            peruse_one_refinement_logic(lines)
        elif MODE == 'final_final':
            lhep = peruse_one_final_final(lines)
            if lhep not in lhep_counter:
                lhep_counter[lhep] = 0

            lhep_counter[lhep] += 1
            for meow in lhep:
                if meow not in meow_counter:
                    meow_counter[meow] = 0

                meow_counter[meow] += 1
        else:
            assert(False)

        print('\n\n')

    if MODE == 'final_final':
        print(lhep_counter)
        print(meow_counter)


if __name__ == '__main__':
    peruse(*(sys.argv[1:]))
