import librosa
import numpy as np

def estimate_pitch(audio_path, sr=16000):
    """
    Estimate the mean pitch (fundamental frequency, F0) of the audio.
    Uses librosa's pyin algorithm to extract pitch per frame.
    Returns mean F0 in Hz, ignoring unvoiced frames.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    mean_f0 = np.nanmean(f0)  # Average pitch ignoring unvoiced (NaN)
    return mean_f0

def estimate_mean_rms_dbfs(audio_path, sr=16000):
    """
    Estimate mean RMS level in decibels relative to full scale (dBFS).
    """
    y, _ = librosa.load(audio_path, sr=sr)
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    mean_rms_dbfs = librosa.amplitude_to_db(np.array([mean_rms]), ref=1.0)[0]
    return mean_rms_dbfs

def audio_properties(audio_path, sr=16000):
    """
    Extract basic audio properties: pitch (Hz) and RMS loudness (dBFS).
    """
    pitch = estimate_pitch(audio_path, sr=sr)
    rms_dbfs = estimate_mean_rms_dbfs(audio_path, sr=sr)

    return {
        "Mean_Pitch_Hz": round(float(pitch), 2),
        "Mean_RMS_dBFS": round(float(rms_dbfs), 2),
    }
