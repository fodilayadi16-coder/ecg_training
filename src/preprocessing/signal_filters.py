"""
ECG signal filtering for preprocessing before training.
========================================================
Removes baseline wander, powerline interference, and high-frequency noise.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, fs=360, lowcut=0.5, highcut=40.0, order=4):
    """
    Bandpass filter to remove baseline wander (< 0.5 Hz)
    and high-frequency noise (> 40 Hz).
    """
    if fs <= 0:
        raise ValueError("fs must be a positive sampling frequency")

    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0)


def notch_filter(signal, fs=360, freq=50.0, quality=30.0):
    """
    Notch filter to remove powerline interference (50 Hz or 60 Hz).
    """
    if fs <= 0:
        raise ValueError("fs must be a positive sampling frequency")

    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal, axis=0)


def normalize_signal(signal):
    """
    Z-score normalization per window.
    """
    std = np.std(signal)
    if std < 1e-6:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def clean_ecg(signal, fs=360, notch_freq=50.0):
    """
    Full cleaning pipeline applied to a single ECG window.

    Steps:
        1. Bandpass 0.5–40 Hz (removes baseline wander + HF noise)
        2. Notch at 50 Hz (removes powerline interference)
        3. Z-score normalization
    """
    sig = signal.copy().astype(np.float64)

    # Handle flat signals
    if np.std(sig) < 1e-6:
        return np.zeros_like(sig, dtype=np.float32)

    sig = bandpass_filter(sig, fs=fs)
    sig = notch_filter(sig, fs=fs, freq=notch_freq)
    sig = normalize_signal(sig)

    return sig.astype(np.float32)