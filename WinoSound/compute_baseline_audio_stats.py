import os
import sys
import json
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from openai import OpenAI
from openai_utils import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
from WinoSound.asr_v2 import whisper_transcribe
from WinoSound.properties_v2 import pitch_stats_aubio
from WinoSound.syllable_counter import count_total_syllables
from WinoSound.make_baseline_audios import VOICES, NUM_REPS, OUT_DIR


#return as dict for json
#stats are mean_pitch (in Hz), mean_syllable_duration (in sec), and mean_speed (in syl/sec)
def compute_stats_one_audio(audio_path):
    mean_pitch, _, __ = pitch_stats_aubio(audio_path)
    transcript, word_chunks = whisper_transcribe(audio_path)
    total_num_syllables = count_total_syllables(transcript)
    total_syllable_time = np.sum([wc[2] - wc[1] for wc in word_chunks])
    mean_syllable_duration = total_syllable_time / total_num_syllables
    total_utterance_time = word_chunks[-1][2] - word_chunks[0][1]
    mean_speed = total_num_syllables / total_utterance_time
    stats_one = {'mean_pitch' : float(mean_pitch), 'mean_syllable_duration' : float(mean_syllable_duration), 'mean_speed' : float(mean_speed)}
    stats_one['total_num_syllables'] = total_num_syllables
    stats_one['total_syllable_time'] = total_syllable_time
    stats_one['total_utterance_time'] = total_utterance_time
    return stats_one


def compute_baseline_audio_stats():
    for voice in VOICES:
        stats = []
        for t in tqdm(range(NUM_REPS)):
            audio_path = os.path.join(OUT_DIR, 'baseline-%s-%02d.wav'%(voice, t))
            stats_one = compute_stats_one_audio(audio_path)
            stats.append(stats_one)

        stats_filename = os.path.join(OUT_DIR, 'baseline-%s-stats.json'%(voice))
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=4)


if __name__ == '__main__':
    compute_baseline_audio_stats()
