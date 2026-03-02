import os
import wfdb
from scipy.signal import resample_poly


def load_record(record_path, lead=0, target_fs=360):
    """Load a WFDB record and its annotations.

    This function normalizes the provided path and wraps the wfdb calls
    with clearer error messages so callers can skip problematic records.
    """
    # Normalize path separators and strip trailing whitespace
    record_path = os.path.normpath(str(record_path)).strip()

    # Try to read record + annotations, raising a descriptive error on failure
    try:
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, 'atr')
    except Exception as e:
        # Re-raise with context so caller knows which file failed
        raise RuntimeError(f"Failed to read WFDB record '{record_path}': {e}") from e

    # Extract signal, annotations and sampling frequency
    try:
        signal = record.p_signal[:, lead]
    except Exception as e:
        raise RuntimeError(f"Unexpected signal format in '{record_path}': {e}") from e

    r_peaks = ann.sample
    symbols = ann.symbol
    fs = float(record.fs)

    # Resample to target_fs if needed
    target_fs = int(target_fs)
    if fs != target_fs:
        signal = resample_poly(signal, target_fs, int(fs))
        r_peaks = (r_peaks * target_fs / fs).astype(int)
        fs = target_fs

    return signal, r_peaks, symbols, fs
