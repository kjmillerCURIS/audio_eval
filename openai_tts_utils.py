import os
import sys
import numpy as np
import librosa
from pydub import AudioSegment
import pyrubberband as pyrb
import torch
import torchaudio
from tqdm import tqdm
import whisperx
from scipy.ndimage import gaussian_filter1d
from openai import OpenAI
from openai_utils import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


client = OpenAI()
TEMP_AUDIO_PATH = 'temp.wav'
device = "cuda" if torch.cuda.is_available() else "cpu"
model_a, metadata = whisperx.load_align_model(language_code="en", device=device)


VOLUME_FACTOR = 9 #3
DURATION_FACTOR = 0.9 #smaller means more stretch
PITCH_FACTOR = 1.0


##start and end should be in seconds
##will return audio, might also modify it in-place
##suggested you do these in reverse so that timing isn't messed up
#def emphasize(audio, start, end):
#    before, during, after = audio[:1000 * start], audio[1000 * start:1000 * end], audio[1000 * end:]
#    during = during + 9
#    #samples = np.array(during.get_array_of_samples()).astype(np.float32) / 32768.0
#    #sr = during.frame_rate
#    ##samples = librosa.effects.time_stretch(samples, rate=0.2)
#    ##samples = librosa.effects.pitch_shift(samples, sr=sr, n_steps=2)
#    #samples = pyrubberband.pyrb.pitch_shift(samples, sr=sr, n_steps=1)
#    #y_int16 = (samples * 32768).astype(np.int16)
#    #during = AudioSegment(y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
#    return before + during + after
#
#
#
#    return before + (during + VOLUME_FACTOR) + after
#    during = during + VOLUME_FACTOR
#    samples = np.array(during.get_array_of_samples()).astype(np.float32) / 32768.0
#    sr = during.frame_rate
#    y_stretched = librosa.effects.time_stretch(samples, rate=DURATION_FACTOR)
#    y_pitched = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=PITCH_FACTOR)
#    y_int16 = (y_pitched * 32768).astype(np.int16)
#    during = AudioSegment(y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
#    return before + during + after


def stretch_common(audio, rate):
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    stretched_samples = pyrb.time_stretch(samples, sr, rate=rate)
    stretched_int16 = (stretched_samples * 32768).astype(np.int16)
    stretched_audio = AudioSegment(
        stretched_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    return stretched_audio


def stretch_subregion_with_context(audio, decoy_audio, start_ms, end_ms, rate=0.9, min_gap_ms=250):
    """
    Time-stretch a short subregion of an AudioSegment using contextual smoothing.
    """
    if end_ms - start_ms < min_gap_ms:
        center_ms = 0.5 * (start_ms + end_ms)
        start_ms, end_ms = max(center_ms - 0.5 * min_gap_ms, 0), center_ms + 0.5 * min_gap_ms

    stretched_audio = stretch_common(audio, rate)

    # --- Step 5: Figure out new region boundaries ---
    new_start_ms, new_end_ms = start_ms / rate, end_ms / rate
    replacement = stretched_audio[new_start_ms:new_end_ms]

#    # --- Step 6: Apply fade-in/out for smoother transition ---
#    if fade_ms > 0:
#        replacement = replacement.fade_in(fade_ms).fade_out(fade_ms)

    # --- Step 7: Stitch back into the original ---
    final_audio = (
        decoy_audio[:start_ms] +
        replacement +
        decoy_audio[end_ms:]
    )

    return final_audio


def apply_gaussian_volume_envelope(audio, start_ms, end_ms, gain_db=6.0, kernel_width_ms=200, step_ms=10, min_gap_ms=250):
    """
    Apply a smooth Gaussian-smoothed volume envelope to an AudioSegment.
    
    Parameters:
        audio: AudioSegment
        start_ms: start of emphasis region
        end_ms: end of emphasis region
        gain_db: max gain during emphasis
        kernel_width_ms: bandwidth of the Gaussian kernel in ms
        step_ms: resolution of volume envelope in ms

    Returns:
        AudioSegment with gain envelope applied.
    """

    #print(len(audio))

    if end_ms - start_ms < min_gap_ms:
        center_ms = 0.5 * (start_ms + end_ms)
        start_ms, end_ms = max(center_ms - 0.5 * min_gap_ms, 0), center_ms + 0.5 * min_gap_ms

    total_ms = len(audio)
    num_steps = total_ms // step_ms
    gain_curve = np.zeros(num_steps)

    # Set gain_db between start and end
    i_start = int(start_ms // step_ms)
    i_end = int(end_ms // step_ms)
    gain_curve[i_start:i_end] = gain_db

    # Smooth the curve with Gaussian
    sigma = kernel_width_ms / step_ms
    smoothed_gain = gaussian_filter1d(gain_curve, sigma=sigma)

    # Apply the gain curve chunk by chunk
    output = AudioSegment.empty()
    for i in range(num_steps):
        t0 = i * step_ms
        t1 = min((i + 1) * step_ms, total_ms)
        chunk = audio[t0:t1]
        chunk = chunk.apply_gain(smoothed_gain[i])
        output += chunk

    # Add any remainder (if total_ms isn't divisible by step_ms)
    if total_ms > num_steps * step_ms:
        chunk = audio[num_steps * step_ms :]
        chunk = chunk.apply_gain(0.0)  # no extra gain
        output += chunk

    #print(len(output))
    return output


#stretch it and unstretch it so the whole thing already has artifacts
def decoy_stretch(audio, decoy_rate=0.95):
    return stretch_common(stretch_common(audio, decoy_rate), 1 / decoy_rate)


def run_tts(text_with_emphasis, emotion, audio_filename):
    if os.path.exists(TEMP_AUDIO_PATH):
        os.remove(TEMP_AUDIO_PATH)

    #TTS
    prefix = 'I am speaking in a %s tone because I am %s.'%(emotion, emotion)
    suffix = 'Again, ' + prefix #mainly meant as padding against whatever is eating up the end sometimes
    spoken_text = prefix + ' ' + text_with_emphasis + ' ' + suffix
    emphasis_indices = [i for i, word in enumerate(spoken_text.split()) if '<emphasis>' in word]
#    print(emphasis_indices)
    first_index_of_payload = len(prefix.split())
    first_index_of_suffix = first_index_of_payload + len(text_with_emphasis.split())
    spoken_text = spoken_text.replace('<emphasis>', '').replace('</emphasis>', '')
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=spoken_text,
        instructions='Speak in a %s tone.'%(emotion),
    ) as response:
        response.stream_to_file(TEMP_AUDIO_PATH)

    #align
    waveform, sample_rate = torchaudio.load(TEMP_AUDIO_PATH)
    duration = waveform.shape[1] / sample_rate
    custom_segment = {
        "start": 0.0,
        "end": duration,
        "text": spoken_text
    }

    alignment = whisperx.align([custom_segment], model_a, metadata, TEMP_AUDIO_PATH, device)
    assert(len(alignment['word_segments']) == len(spoken_text.split()))

    #add emphasis
    audio = AudioSegment.from_file(TEMP_AUDIO_PATH)
#    decoy_audio = decoy_stretch(audio)
    for index in tqdm(emphasis_indices[::-1]):
        #print(alignment['word_segments'][index]['word'])
        start, end = alignment['word_segments'][index]['start'], alignment['word_segments'][index]['end']
        #print((start, end))
        #audio = emphasize(audio, start, end)
        audio = apply_gaussian_volume_envelope(audio, start * 1000, end * 1000, gain_db=12.0, kernel_width_ms=50, step_ms=5)
        #audio = stretch_subregion_with_context(audio, decoy_audio, start * 1000, end * 1000, rate=0.5)

#    for index in tqdm(emphasis_indices[::-1]):
#        print(alignment['word_segments'][index]['word'])
#        start, end = alignment['word_segments'][index]['start'], alignment['word_segments'][index]['end']
#        print((start, end))
#        #audio = emphasize(audio, start, end)
#        #audio = apply_gaussian_volume_envelope(audio, start * 1000, end * 1000, gain_db=18.0, kernel_width_ms=50, step_ms=5)
#        audio = stretch_subregion_with_context(audio, decoy_audio, start * 1000, end * 1000, rate=0.5)

    #crop
    crop_start_ms = 500 * (alignment['word_segments'][first_index_of_payload]['start'] + alignment['word_segments'][first_index_of_payload-1]['end'])
    crop_end_ms = 500 * (alignment['word_segments'][first_index_of_suffix]['start'] + alignment['word_segments'][first_index_of_suffix-1]['end'])
    audio = audio[crop_start_ms:crop_end_ms]

    #save
    audio.export(audio_filename, format=os.path.splitext(audio_filename)[-1].replace('.', ''))


if __name__ == '__main__':
    text_with_emphasis = 'Why the heck <emphasis>am</emphasis> I so damn <emphasis>tired</emphasis>?'
    emotion = 'frustrated'
    run_tts(text_with_emphasis, emotion, 'tired.wav')
