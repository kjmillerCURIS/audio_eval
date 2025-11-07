import os
import sys
import json
import math
import numpy as np
from tqdm import tqdm
sys.path.append('.')
from openai import OpenAI
from openai_utils import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
from WinoSound.asr_v2 import whisper_transcribe
from WinoSound.syllable_counter import count_total_syllables
from WinoSound.asr_v2 import whisper_transcribe
from WinoSound.properties_v2 import pitch_stats_aubio
from WinoSound.emotion import emotion_to_vec_scores
from WinoSound.syllable_counter import count_total_syllables
from WinoSound.make_baseline_audios import OUT_DIR as BASELINE_DIR


EMOTION2VEC_EMOTIONS = ['neutral', 'happy', 'sad', 'surprised', 'fearful', 'angry', 'disgusted']
SEMITONE_RATIO = 1.05946
PITCH_END_DURATION = 0.6
PITCH_END_NUM_POINTS = 7
HESITANT_SHORT_MODE_WORD_THRESHOLD = 5
HESITANT_END_PITCH_SLOPE_THRESHOLD = 6.0
HESITANT_PAUSE_SYLLABLES_RANK_PROP = 0.25
HESITANT_PAUSE_SYLLABLES_MAX_RANK = 2
HESITANT_PAUSE_SYLLABLES_THRESHOLD = 2.0
FRAZZLED_SHORT_MODE_SYLLABLE_THRESHOLD = 2
FRAZZLED_RELATIVE_SPEED_THRESHOLD = 0.3
FRAZZLED_RELATIVE_PITCH_THRESHOLD = 5.0
IMPATIENT_SHORT_MODE_SYLLABLE_THRESHOLD = 2
IMPATIENT_RELATIVE_SPEED_THRESHOLD = 0.3
IMPATIENT_RELATIVE_PITCH_THRESHOLD = 4.0
EMPATHETIC_ANGRY_DISGUSTED_THRESHOLD = 0.01
EMPATHETIC_RELATIVE_SPEED_THRESHOLD = 0.02
BORED_SCORE_THRESHOLD = 0.05
ANGRY_SCORE_THRESHOLD = 0.5
NEUTRAL_SCORE_THRESHOLD = 0.8
SAD_SCORE_THRESHOLD = 0.03
HAPPY_SCORE_THRESHOLD = 0.005


def compute_end_pitch_slope(audio_path, word_chunks):
    T = np.sum([wc[2] - wc[1] for wc in word_chunks])
    if T < PITCH_END_DURATION:
        contour_length = PITCH_END_NUM_POINTS
    else:
        contour_length = int(math.ceil((T / PITCH_END_DURATION) * PITCH_END_NUM_POINTS))

    _, __, pitch_contour = pitch_stats_aubio(audio_path, contour_length=contour_length)
    pitch_contour = np.log(pitch_contour) / np.log(SEMITONE_RATIO)
    end_contour = pitch_contour[-PITCH_END_NUM_POINTS:]
    times = np.linspace(0, (PITCH_END_NUM_POINTS / contour_length) * T, PITCH_END_NUM_POINTS)
    xdiffs = times - np.mean(times)
    ydiffs = end_contour - np.mean(end_contour)
    slope = np.sum(xdiffs * ydiffs) / np.sum(np.square(xdiffs))
    #print(pitch_contour)
    #print(end_contour)
    #print(slope)
    #print('')
    return slope


#whisper tends to ignore pauses, so we "fake" some by assuming that each syllable takes the baseline duration and timing the words accordingly
#(won't use these for pitch slope computation, just for pause-based metrics)
def revise_word_chunks(word_chunks, syllable_duration):
    revised_word_chunks = []
    for wc in word_chunks:
        c = 0.5 * (wc[1] + wc[2])
        n = count_total_syllables(wc[0])
        a = c - 0.5 * n * syllable_duration
        b = c + 0.5 * n * syllable_duration
        revised_word_chunks.append((wc[0], a, b))

    return revised_word_chunks


def score_hesitant(audio_path, voice, name):
    extra = {}
    baseline_stats = load_baseline_stats(audio_path, voice, name)
    syllable_duration = np.mean([s['mean_syllable_duration'] for s in baseline_stats])
    transcript, word_chunks = whisper_transcribe(audio_path)
    end_pitch_slope = compute_end_pitch_slope(audio_path, word_chunks)
    extra['end_pitch_slope'] = end_pitch_slope
    qualified = int(end_pitch_slope >= HESITANT_END_PITCH_SLOPE_THRESHOLD)
    extra['num_words'] = len(word_chunks)
    if len(word_chunks) >= HESITANT_SHORT_MODE_WORD_THRESHOLD:
        revised_word_chunks = revise_word_chunks(word_chunks, syllable_duration)
        pauses = [(revised_word_chunks[i + 1][1] - revised_word_chunks[i][2]) / syllable_duration for i in range(len(revised_word_chunks) - 1)]
        sorted_pauses = sorted(pauses, reverse=True)
        extra['sorted_pauses'] = sorted_pauses
        rank = int(round(HESITANT_PAUSE_SYLLABLES_RANK_PROP * len(pauses)))
        rank = min(max(rank, 1), HESITANT_PAUSE_SYLLABLES_MAX_RANK)
        extra['pause_rank'] = rank
        kth_longest_pause = sorted_pauses[rank - 1]
        extra['kth_longest_pause'] = kth_longest_pause
        qualified *= int(kth_longest_pause >= HESITANT_PAUSE_SYLLABLES_THRESHOLD)

    extra['qualified'] = qualified
    score = end_pitch_slope
    extra['score'] = score
    return qualified, score, extra


def load_baseline_stats(audio_path, voice, name):
    if voice is not None:
        stats_filename = os.path.join(os.path.dirname(audio_path), '%s-%s-neutral-stats.json'%(name, voice))
    else:
        stats_filename = '%s/%s-neutral-stats.json'%(os.path.dirname(audio_path), name)

    with open(stats_filename, 'r') as f:
        baseline_stats = json.load(f)

    return baseline_stats


def compute_relative_speed(word_chunks, baseline_stats):
    baseline_speed = np.mean([s['mean_speed'] for s in baseline_stats])
    total_syllables = np.sum([int(count_total_syllables(wc[0])) for wc in word_chunks])
    total_utterance_time = float(word_chunks[-1][2] - word_chunks[0][1])
    my_speed = total_syllables / total_utterance_time
    return (my_speed - baseline_speed) / baseline_speed


def compute_relative_pitch(audio_path, baseline_stats):
    baseline_pitch = np.mean([s['mean_pitch'] for s in baseline_stats])
    my_pitch, _, __ = pitch_stats_aubio(audio_path)
    return (np.log(float(my_pitch)) - np.log(baseline_pitch)) / np.log(SEMITONE_RATIO)


def score_frazzled(audio_path, voice, name):
    extra = {}
    baseline_stats = load_baseline_stats(audio_path, voice, name)
    transcript, word_chunks = whisper_transcribe(audio_path)
    relative_pitch = float(compute_relative_pitch(audio_path, baseline_stats))
    extra['relative_pitch'] = float(relative_pitch)
    qualified = int(relative_pitch >= FRAZZLED_RELATIVE_PITCH_THRESHOLD)
    total_syllables = count_total_syllables(transcript)
    extra['total_syllables'] = int(total_syllables)
    if total_syllables >= FRAZZLED_SHORT_MODE_SYLLABLE_THRESHOLD:
        relative_speed = float(compute_relative_speed(word_chunks, baseline_stats))
        extra['relative_speed'] = float(relative_speed)
        qualified *= (relative_speed >= FRAZZLED_RELATIVE_SPEED_THRESHOLD)

    extra['qualified'] = qualified
    score = float(relative_pitch)
    extra['score'] = score
    return qualified, score, extra


def score_impatient(audio_path, voice, name):
    extra = {}
    baseline_stats = load_baseline_stats(audio_path, voice, name)
    transcript, word_chunks = whisper_transcribe(audio_path)
    relative_pitch = float(compute_relative_pitch(audio_path, baseline_stats))
    extra['relative_pitch'] = float(relative_pitch)
    qualified = int(np.fabs(relative_pitch) <= IMPATIENT_RELATIVE_PITCH_THRESHOLD)
    total_syllables = count_total_syllables(transcript)
    extra['total_syllables'] = int(total_syllables)
    if total_syllables >= IMPATIENT_SHORT_MODE_SYLLABLE_THRESHOLD:
        relative_speed = float(compute_relative_speed(word_chunks, baseline_stats))
        extra['relative_speed'] = float(relative_speed)
        qualified *= (relative_speed >= IMPATIENT_RELATIVE_SPEED_THRESHOLD)

    extra['qualified'] = qualified
    score = -float(np.fabs(relative_pitch))
    extra['score'] = score
    return qualified, score, extra


def compute_normalized_emotion2vec(audio_path):
    orig_vec = emotion_to_vec_scores(audio_path)
    vec = {}
    total = np.sum([orig_vec[emotion] for emotion in EMOTION2VEC_EMOTIONS])
    for emotion in EMOTION2VEC_EMOTIONS:
        vec[emotion] = orig_vec[emotion] / total

    return vec


def score_empathetic(audio_path, voice, name):
    extra = {}
    vec = compute_normalized_emotion2vec(audio_path)
    extra['emotion2vec_components'] = {emotion : vec[emotion] for emotion in ['angry', 'disgusted']}
    qualified = int(vec['angry'] + vec['disgusted'] <= EMPATHETIC_ANGRY_DISGUSTED_THRESHOLD)
    baseline_stats = load_baseline_stats(audio_path, voice, name)
    transcript, word_chunks = whisper_transcribe(audio_path)
    relative_speed = float(compute_relative_speed(word_chunks, baseline_stats))
    extra['relative_speed'] = relative_speed
    qualified *= int(relative_speed <= EMPATHETIC_RELATIVE_SPEED_THRESHOLD)
    score = -relative_speed
    extra['score'] = score
    extra['qualified'] = qualified
    return qualified, score, extra


def score_bored(audio_path):
    extra = {}
    vec = compute_normalized_emotion2vec(audio_path)
    extra['emotion2vec_components'] = {emotion : vec[emotion] for emotion in ['neutral', 'sad', 'disgusted']}
    score = np.sqrt(vec['neutral'] * (vec['sad'] + vec['disgusted']))
    qualified = int(score >= BORED_SCORE_THRESHOLD)
    extra['score'] = score
    extra['qualified'] = qualified
    return qualified, score, extra


def score_angry(audio_path):
    extra = {}
    vec = compute_normalized_emotion2vec(audio_path)
    extra['emotion2vec_components'] = {emotion : vec[emotion] for emotion in ['angry', 'disgusted']}
    score = vec['angry'] + vec['disgusted']
    qualified = int(score >= ANGRY_SCORE_THRESHOLD)
    extra['score'] = score
    extra['qualified'] = qualified
    return qualified, score, extra


def score_neutral(audio_path):
    extra = {}
    vec = compute_normalized_emotion2vec(audio_path)
    extra['emotion2vec_components'] = {emotion : vec[emotion] for emotion in ['neutral']}
    score = vec['neutral']
    qualified = int(score >= NEUTRAL_SCORE_THRESHOLD)
    extra['score'] = score
    extra['qualified'] = qualified
    return qualified, score, extra


def score_sad(audio_path):
    extra = {}
    vec = compute_normalized_emotion2vec(audio_path)
    extra['emotion2vec_components'] = {emotion : vec[emotion] for emotion in ['sad']}
    score = vec['sad']
    qualified = int(score >= SAD_SCORE_THRESHOLD)
    extra['score'] = score
    extra['qualified'] = qualified
    return qualified, score, extra


def score_happy(audio_path):
    extra = {}
    vec = compute_normalized_emotion2vec(audio_path)
    extra['emotion2vec_components'] = {emotion : vec[emotion] for emotion in ['happy']}
    score = vec['happy']
    qualified = int(score >= HAPPY_SCORE_THRESHOLD)
    extra['score'] = score
    extra['qualified'] = qualified
    return qualified, score, extra


class TTSScorer:
    def __init__(self):
        pass

    def score(self, audio_path, target_emotion, voice, name):
        if target_emotion == 'hesitant':
            qualified, score, extra = score_hesitant(audio_path, voice, name)
        elif target_emotion == 'frazzled':
            qualified, score, extra = score_frazzled(audio_path, voice, name)
        elif target_emotion == 'impatient':
            qualified, score, extra = score_impatient(audio_path, voice, name)
        elif target_emotion == 'empathetic':
            qualified, score, extra = score_empathetic(audio_path, voice, name)
        elif target_emotion == 'bored':
            qualified, score, extra = score_bored(audio_path)
        elif target_emotion == 'angry':
            qualified, score, extra = score_angry(audio_path)
        elif target_emotion == 'neutral':
            qualified, score, extra = score_neutral(audio_path)
        elif target_emotion == 'sad':
            qualified, score, extra = score_sad(audio_path)
        elif target_emotion == 'happy':
            qualified, score, extra = score_happy(audio_path)
        else:
            assert(False)

        return qualified, score, extra


def main():
    my_scorer = TTSScorer()
    for name in ['long', 'noflights', 'hawaiiflight', 'hello', 'hi']:
        for voice in ['echo', 'alloy', 'ash']:
            for target_emotion in ['hesitant', 'frazzled', 'impatient', 'empathetic', 'bored', 'angry', 'neutral', 'sad', 'happy']:
                extras = {}
                for t in tqdm(range(10)):
                    audio_path = 'WinoSound/tts_samples/%s-%s-%s-%02d.wav'%(name, voice, target_emotion, t)
                    #print(os.path.basename(audio_path))
                    qualified, score, extra = my_scorer.score(audio_path, target_emotion, voice, name)
                    print('%s: qualified=%d, score=%f'%(os.path.basename(audio_path), qualified, score))
                    print(extra)
                    extras[os.path.basename(audio_path)] = extra

                extras_filename = 'WinoSound/tts_samples/%s-%s-%s-extras.json'%(name, voice, target_emotion)
                with open(extras_filename, 'w') as f:
                    json.dump(extras, f, indent=4)


if __name__ == '__main__':
    main()
