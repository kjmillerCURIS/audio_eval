import librosa
import numpy as np

def estimate_pitch(audio_path, sr=16000):
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
    mean_pitch, std_dev_pitch = estimate_pitch(audio_path, sr=sr)
    rms_dbfs = estimate_mean_rms_dbfs(audio_path, sr=sr)

    return {
        "Mean_Pitch_Hz": round(float(mean_pitch), 2),
        "Std_Dev_Pitch_Hz": round(float(std_dev_pitch), 2),
        "Mean_RMS_dBFS": round(float(rms_dbfs), 2),
    }
