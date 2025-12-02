import os
import sys
import glob
from tqdm import tqdm
import soundfile as sf


def convert_one_file(before_filename, after_filename):
    data, samplerate = sf.read(before_filename)
    sf.write(after_filename, data, samplerate, format="WAV")


def convert_audio_files(parent_dir):
    filenames = sorted(glob.glob(os.path.join(parent_dir, '*', '*.wav')))
    for filename in tqdm(filenames):
        convert_one_file(filename, filename)


if __name__ == '__main__':
    convert_audio_files(*(sys.argv[1:]))
