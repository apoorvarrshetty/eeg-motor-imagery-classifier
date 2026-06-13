"""
preprocess.py
-------------
Load PhysioNet Motor Imagery dataset, epoch it, and extract band-power features.
"""

import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf


# ── 1. Constants ────────────────────────────────────────────────────────────

SUBJECT    = 1           # PhysioNet subject ID (1–109)
RUNS_LEFT_RIGHT = [6, 10, 14]   # runs with left/right hand imagery
TMIN, TMAX = 0.0, 4.0   # epoch window in seconds
FREQ_BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}


# ── 2. Data loading ──────────────────────────────────────────────────────────

def load_raw(subject: int = SUBJECT, runs: list = RUNS_LEFT_RIGHT) -> mne.io.Raw:
    """Download (if needed) and return concatenated raw EEG for one subject."""
    files = eegbci.load_data(subject, runs, path="data/")
    raws  = [read_raw_edf(f, preload=True, stim_channel="auto") for f in files]
    raw   = concatenate_raws(raws)

    # Standardise channel names to 10-20 montage
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)

    return raw


# ── 3. Preprocessing ─────────────────────────────────────────────────────────

def preprocess(raw: mne.io.Raw) -> mne.io.Raw:
    """Band-pass filter and apply common-average reference."""
    raw.filter(1.0, 45.0, fir_design="firwin")
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()
    return raw


# ── 4. Epoching ──────────────────────────────────────────────────────────────

def make_epochs(raw: mne.io.Raw) -> mne.Epochs:
    """
    Create epochs time-locked to motor-imagery onset.
    Labels:  T1 → left hand (0),  T2 → right hand (1)
    """
    events, event_id = mne.events_from_annotations(raw)

    # Keep only T1 (left) and T2 (right)
    epochs = mne.Epochs(
        raw, events,
        event_id={"T1": event_id["T1"], "T2": event_id["T2"]},
        tmin=TMIN, tmax=TMAX,
        baseline=None, preload=True
    )
    return epochs


# ── 5. Feature extraction ────────────────────────────────────────────────────

def bandpower(data: np.ndarray, sfreq: float, band: tuple) -> np.ndarray:
    """
    Compute average band power for each channel using Welch's method.

    Parameters
    ----------
    data  : (n_channels, n_times)
    sfreq : sampling frequency in Hz
    band  : (low_freq, high_freq)

    Returns
    -------
    power : (n_channels,)
    """
    from scipy.signal import welch

    fmin, fmax = band
    freqs, psd = welch(data, sfreq, nperseg=sfreq)          # PSD per channel
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return psd[:, idx].mean(axis=-1)                         # mean power in band


def extract_features(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """
    For every epoch, concatenate band-power features across all channels and bands.

    Returns
    -------
    X : (n_epochs, n_channels * n_bands)  — feature matrix
    y : (n_epochs,)                        — 0 = left, 1 = right
    """
    data   = epochs.get_data()          # (n_epochs, n_channels, n_times)
    sfreq  = epochs.info["sfreq"]
    labels = (epochs.events[:, -1] == epochs.event_id["T2"]).astype(int)

    features = []
    for epoch in data:                  # epoch shape: (n_channels, n_times)
        epoch_feats = []
        for band in FREQ_BANDS.values():
            bp = bandpower(epoch, sfreq, band)   # (n_channels,)
            epoch_feats.append(bp)
        features.append(np.concatenate(epoch_feats))

    X = np.array(features)
    y = labels
    print(f"Feature matrix: {X.shape}  |  Labels: {np.bincount(y)}")
    return X, y


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raw    = load_raw()
    raw    = preprocess(raw)
    epochs = make_epochs(raw)
    X, y   = extract_features(epochs)
    print("Done. Ready for classification.")
