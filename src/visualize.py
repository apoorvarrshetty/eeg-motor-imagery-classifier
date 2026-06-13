"""
visualize.py
------------
Helper functions to plot raw EEG, PSD, and epoch topomaps.
Run standalone to generate all plots for a quick sanity check.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne


def plot_raw_snippet(raw: mne.io.Raw, duration: float = 5.0, n_channels: int = 8) -> None:
    """Plot the first `duration` seconds of `n_channels` channels."""
    sfreq    = raw.info["sfreq"]
    n_times  = int(duration * sfreq)
    data, times = raw[:n_channels, :n_times]

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 1.2), sharex=True)
    ch_names  = raw.ch_names[:n_channels]

    for i, (ax, ch) in enumerate(zip(axes, ch_names)):
        ax.plot(times, data[i] * 1e6, lw=0.8, color="#1f77b4")  # convert V → µV
        ax.set_ylabel(ch, fontsize=7, rotation=0, labelpad=35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw EEG — First 5 seconds", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("results/raw_eeg_snippet.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → results/raw_eeg_snippet.png")


def plot_psd(raw: mne.io.Raw) -> None:
    """Plot power spectral density (1–45 Hz) averaged across channels."""
    fig = raw.compute_psd(fmin=1, fmax=45).plot(show=False)
    fig.suptitle("Power Spectral Density", fontsize=11)
    fig.savefig("results/psd.png", dpi=150)
    plt.close(fig)
    print("Saved → results/psd.png")


def plot_band_power_comparison(epochs: mne.Epochs) -> None:
    """
    Bar chart: mean alpha & beta power for left-hand vs right-hand epochs
    averaged across all channels.
    """
    from scipy.signal import welch

    sfreq  = epochs.info["sfreq"]
    bands  = {"Alpha (8-13 Hz)": (8, 13), "Beta (13-30 Hz)": (13, 30)}

    # Map each event id to a friendly label. T1 = left hand, T2 = right hand.
    pretty = {"T1": "T1 (Left)", "T2": "T2 (Right)"}
    labels = {v: pretty.get(str(k), str(k)) for k, v in epochs.event_id.items()}
    groups = {name: [] for name in labels.values()}

    for epoch, event in zip(epochs.get_data(), epochs.events[:, -1]):
        tag = labels[event]
        for bname, (flo, fhi) in bands.items():
            freqs, psd = welch(epoch, sfreq, nperseg=int(sfreq))
            idx  = (freqs >= flo) & (freqs <= fhi)
            power = psd[:, idx].mean()
            groups[tag].append((bname, power))

    # Aggregate
    summary = {}
    for tag, entries in groups.items():
        for bname, p in entries:
            summary.setdefault(bname, {})[tag] = summary.get(bname, {}).get(tag, [])
            summary[bname][tag].append(p)

    bnames = list(bands.keys())
    tags   = list(groups.keys())
    x      = np.arange(len(bnames))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, tag in enumerate(tags):
        means = [np.mean(summary[b][tag]) for b in bnames]
        ax.bar(x + i * width, means, width, label=tag,
               color=["#4C72B0", "#DD8452"][i])

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(bnames)
    ax.set_ylabel("Mean Band Power (µV²/Hz)")
    ax.set_title("Alpha & Beta Power: Left vs Right Hand Imagery")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/band_power_comparison.png", dpi=150)
    plt.close()
    print("Saved → results/band_power_comparison.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_raw, preprocess, make_epochs

    os.makedirs("results", exist_ok=True)

    raw    = load_raw()
    raw    = preprocess(raw)
    epochs = make_epochs(raw)

    plot_raw_snippet(raw)
    plot_psd(raw)
    plot_band_power_comparison(epochs)
    print("\nAll plots saved to results/")