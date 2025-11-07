import os
import sys
import json
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from WinoSound.compute_baseline_audio_stats import compute_stats_one_audio, VOICES, NUM_REPS


OUT_DIR = 'WinoSound/tts_samples'


def compute_neutral_audio_stats_text_specific(name):
    for voice in VOICES:
        stats = []
        for t in tqdm(range(NUM_REPS)):
            audio_path = os.path.join(OUT_DIR, '%s-%s-neutral-%02d.wav'%(name, voice, t))
            stats_one = compute_stats_one_audio(audio_path)
            stats.append(stats_one)

        stats_filename = os.path.join(OUT_DIR, '%s-%s-neutral-stats.json'%(name, voice))
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=4)


def usage():
    print('Usage: python compute_neutral_audio_stats_text_specific.py <name>')


if __name__ == '__main__':
    compute_neutral_audio_stats_text_specific(*(sys.argv[1:]))
