import os
import sys
import glob
from tqdm import tqdm
import soundfile as sf


def convert_one_file(before_filename, after_filename):
    assert(before_filename != after_filename)
    data, samplerate = sf.read(before_filename)
    sf.write(after_filename, data, samplerate, format="WAV")


def convert_audio_files_meow(input_or_output):
    assert(input_or_output in ['input', 'output'])
    if input_or_output == 'input':
        search_strs = ['json_as_a_judge/s2sarena_experiments/audio_files/%s/*/*.*'%(input_or_output), 'json_as_a_judge/s2sarena_experiments/audio_files/%s/*/*/*.*'%(input_or_output), 'json_as_a_judge/s2sarena_experiments/audio_files/%s/*/*/*/*.*'%(input_or_output), 'json_as_a_judge/s2sarena_experiments/audio_files/%s/*/*/*/*/*.*'%(input_or_output)]
    else:
        search_strs = ['json_as_a_judge/s2sarena_experiments/audio_files/%s/*/*/*.*'%(input_or_output), 'json_as_a_judge/s2sarena_experiments/audio_files/%s/*/*/*/*.*'%(input_or_output), 'json_as_a_judge/s2sarena_experiments/audio_files/%s/*/*/*/*/*.*'%(input_or_output), 'json_as_a_judge/s2sarena_experiments/audio_files/%s/*/*/*/*/*/*.*'%(input_or_output)]

    before_filenames = []
    for search_str in search_strs:
        before_filenames.extend(sorted(glob.glob(search_str)))

    print(set([os.path.splitext(x)[-1] for x in before_filenames]))
    for before_filename in tqdm(before_filenames):
        after_filename = before_filename.replace('audio_files/%s'%(input_or_output), 'audio_files/%s_converted'%(input_or_output))
        os.makedirs(os.path.dirname(after_filename), exist_ok=True)
        convert_one_file(before_filename, after_filename)


def convert_audio_files():
    convert_audio_files_meow('input')
    convert_audio_files_meow('output')


if __name__ == '__main__':
    convert_audio_files(*(sys.argv[1:]))
