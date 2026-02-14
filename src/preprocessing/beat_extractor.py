import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) # include the src directory in the path

from loaders.beat_rec_finder import get_all_records
from loaders.beat_wfdb_loader import load_record
from preprocessing.rr_intervals import plot_rr_distribution
from preprocessing.label_maps import beat_label_map
from preprocessing.signal_filters import clean_ecg


# ----------------------------------------------- Extract Beats from All Records -----------------------------------------------

def extract_beats(records, window_size=180, collect_rr=True):  # 0.5 seconds segmented window (generally RR interval = 0.6 to 1 second)
    half_window = window_size // 2
    X = []              # input features: beat signals (not suitable for CNN models)
    y = []              # labels
    rr_intervals_all = []

    for rec_path in records:
        try:
            signal, r_peaks, symbols, fs = load_record(rec_path)

                # ---------- RR INTERVAL COLLECTION ----------

            if collect_rr and len(r_peaks) > 1:
                rr = np.diff(r_peaks) / fs  # seconds
                rr_intervals_all.extend(rr)

                # ---------- BEAT EXTRACTION ----------

            for i, r in enumerate(r_peaks):
                if symbols[i] not in beat_label_map:
                    continue
                if r - half_window >= 0 and r + half_window < len(signal):   # Checks if the window fits within the signal bounds.
                    beat = signal[r - half_window : r + half_window]
                    beat = clean_ecg(beat, fs=fs)  # bandpass + notch + z-score
                    X.append(beat)
                    y.append(beat_label_map[symbols[i]])

        except Exception as e:
            print("Skipped:", rec_path, e)

    X = np.array(X).reshape(-1, window_size, 1) # 3D array == Tensor of beat signals (shape: (num_beats, 187, 1))
    y = np.array(y)                             # 1D array of shape (beat_labels,)  .eg (1000,)
    rr_intervals_all = np.array(rr_intervals_all)                        

    return X, y, rr_intervals_all


# ----------------------------------------------- Main Script -----------------------------------------------

if __name__ == "__main__":
    records = get_all_records("data/raw/beat")
    print("Total records found:", len(records))

    X, y, rr_intervals  = extract_beats(records)
    print("Beats extracted:", X.shape[0])

    # Save processed dataset
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/beat_X.npy", X)
    np.save("data/processed/beat_y.npy", y)

    print("Saved beat dataset to data/processed/")
 
    plot_rr_distribution(rr_intervals) # Normal RR intervals: 0.6-1s
