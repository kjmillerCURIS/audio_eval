import librosa
import numpy as np
import aubio
import soundfile as sf
import pyloudnorm as pyln


def pitch_stats_librosa(audio_path, sr=16000):
    """
    Estimate the mean and standard deviation of pitch (fundamental frequency, F0) in the audio.
    Uses librosa's pyin algorithm to extract pitch per frame.
    Ignores unvoiced (NaN) frames.

    Returns:
        mean_f0 (float): Mean pitch in Hz.
        std_f0 (float): Standard deviation of pitch in Hz.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)

    # Only use voiced frames (non-NaN)
    voiced_f0 = f0[~np.isnan(f0)]

    mean_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else float('nan')
    std_f0 = np.std(voiced_f0) if len(voiced_f0) > 0 else float('nan')

    return mean_f0, std_f0


#(min_pitch used to be 80, max_pitch used to be 300)
def pitch_stats_aubio(audio_path, samplerate=24000, hop_size=512, method="yin",
                              min_pitch=70.0, max_pitch=500.0, contour_length=20):
    """
    Extract pitch using Aubio, filter by range, return mean/std and fixed-length interpolated contour.
    Returns:
        mean_pitch (float): Mean pitch in Hz.
        std_pitch (float): Standard deviation of pitch.
        contour (np.ndarray): Interpolated pitch contour of length `contour_length`.
    """
    # 1. Load and resample audio
    y, _ = librosa.load(audio_path, sr=samplerate)

    # 2. Setup Aubio pitch detector
    win_s = 2048
    pitch_o = aubio.pitch(method, win_s, hop_size, samplerate)
    pitch_o.set_unit("Hz")
    pitch_o.set_silence(-40)

    # 3. Extract pitches
    pitches = []
    for i in range(0, len(y) - hop_size, hop_size):
        frame = y[i:i+hop_size].astype(np.float32)
        pitch_val = pitch_o(frame)[0]
        confidence = pitch_o.get_confidence()
        if np.isnan(pitch_val) or confidence < 0.8:
            continue
        if min_pitch <= pitch_val <= max_pitch:
            pitches.append(pitch_val)

    pitches = np.array(pitches)

    # 4. Compute mean and std
    if len(pitches) == 0:
        mean_pitch, std_pitch = float('nan'), float('nan')
        contour = np.full(contour_length, np.nan)
    else:
        mean_pitch = np.mean(pitches)
        std_pitch = np.std(pitches)

        # 5. Interpolated contour
        x_orig = np.linspace(0, 1, len(pitches))
        x_new = np.linspace(0, 1, contour_length)
        contour = np.interp(x_new, x_orig, pitches)

    return mean_pitch, std_pitch, contour


def estimate_mean_rms_dbfs(audio_path, sr=16000):
    """
    - Estimate mean RMS level in decibels relative to full scale (dBFS).
    - This may not be a great measure of perceived loudness
    """
    y, _ = librosa.load(audio_path, sr=sr)
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    mean_rms_dbfs = librosa.amplitude_to_db(np.array([mean_rms]), ref=1.0)[0]
    return mean_rms_dbfs

def loudness_stats_lufs(audio_path, contour_length=20):
    """
    - Industry standard for measuring percieved loudness (LUFS)
    - Modified pyloudnorm to get momentary loudness: https://www.izotope.com/en/learn/what-are-lufs
    """
    data, rate = sf.read(audio_path) # load audio (with shape (samples, channels))
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness, loudness_contour = meter.integrated_loudness(data, return_contour=True) # measure loudness

    loudness_stdev = np.std(loudness_contour)

    # Interpolated contour
    x_orig = np.linspace(0, 1, len(loudness_contour))
    x_new = np.linspace(0, 1, contour_length)
    contour_interp = np.interp(x_new, x_orig, loudness_contour)

    return loudness, loudness_stdev, contour_interp


def calculate_speech_rate(word_chunks, audio_path):
    """
    Calculate speech rate (WPM) including pauses.
    """
    num_words = len(word_chunks)
    duration = librosa.get_duration(path=audio_path)
    wpm = (num_words / duration) * 60 
    return round(wpm, 2)
    
def calculate_articulation_rate(word_chunks):
    num_words = len(word_chunks)
    speech_duration = sum(end - start for _, start, end in word_chunks)
    articulation_rate = (num_words / speech_duration) * 60
    return round(articulation_rate, 2)

def audio_properties(audio_path, word_chunks):
    """
    Extract basic audio properties: pitch (Hz) and RMS loudness (dBFS).
    """
    mean_pitch, std_dev_pitch, pitch_contour = pitch_stats_aubio(audio_path)
    integrated_loudness, std_dev_lufs, contour_lufs = loudness_stats_lufs(audio_path)
    speech_rate = calculate_speech_rate(word_chunks, audio_path)
    articulation_rate = calculate_articulation_rate(word_chunks)

    return {
        "Mean_Pitch_Hz": round(float(mean_pitch), 2),
        "Std_Dev_Pitch_Hz": round(float(std_dev_pitch), 2),
        "Full_Pitch_Contour_Hz": np.round(pitch_contour, 2).tolist(),
        "Integrated_Loudness_LUFS": round(float(integrated_loudness), 2),
        "Std_Dev_Loudness_LUFS": round(float(std_dev_lufs), 2),
        "Full_Loudness_Contour_LUFS": np.round(contour_lufs, 2).tolist(),
        "Speech_Rate_WPM": speech_rate,
        "Articulation_Rate_WPM": articulation_rate,
    }

