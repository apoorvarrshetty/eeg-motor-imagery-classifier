"""
app.py — Streamlit frontend for the EEG Motor Imagery Classifier
----------------------------------------------------------------
Pick a subject, run the full pipeline live, and see the results in the browser:
raw EEG, band-power comparison, classifier accuracy, and a confusion matrix.

Run with:   streamlit run app.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Make the src/ modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import (
    load_raw, preprocess, make_epochs, extract_features, FREQ_BANDS
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.signal import welch


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="EEG Motor Imagery Classifier", page_icon="🧠", layout="wide")

st.title("🧠 EEG Motor Imagery Classifier")
st.markdown(
    "Reads brainwaves to guess whether a person imagined moving their "
    "**left hand** or **right hand** — a simple brain-computer interface demo."
)


# ── Cached pipeline (so re-runs are fast) ──────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(subject: int):
    """Run the full EEG pipeline for one subject. Cached per subject."""
    raw    = load_raw(subject=subject)
    raw    = preprocess(raw)
    epochs = make_epochs(raw)
    X, y   = extract_features(epochs)

    # Train an SVM and evaluate on a held-out split
    clf = Pipeline([("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    # A more honest number: average accuracy over several random splits.
    # (Simple loop instead of cross_val_score — fast and robust on small data.)
    accs = []
    for seed in range(5):
        xa, xb, ya, yb = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=seed)
        c = Pipeline([("scaler", StandardScaler()),
                      ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))])
        c.fit(xa, ya)
        accs.append(c.score(xb, yb))
    accs = np.array(accs)

    cm = confusion_matrix(y_te, y_pred)

    # Bundle everything the UI needs (arrays only — figures built outside cache)
    return {
        "raw_data":   raw.get_data(picks=range(min(8, len(raw.ch_names))),
                                   start=0, stop=int(5 * raw.info["sfreq"])),
        "raw_times":  np.arange(int(5 * raw.info["sfreq"])) / raw.info["sfreq"],
        "ch_names":   raw.ch_names[:8],
        "epochs_data": epochs.get_data(),
        "epoch_events": epochs.events[:, -1],
        "event_id":   {str(k): v for k, v in epochs.event_id.items()},
        "sfreq":      epochs.info["sfreq"],
        "n_epochs":   len(y),
        "n_left":     int(np.sum(y == 0)),
        "n_right":    int(np.sum(y == 1)),
        "test_acc":   clf.score(X_te, y_te),
        "cv_mean":    accs.mean(),
        "cv_std":     accs.std(),
        "cm":         cm,
    }


@st.cache_data(show_spinner=False)
def quick_accuracy(subject: int) -> float:
    """Lightweight: return only the mean accuracy for a subject (for comparison)."""
    raw    = preprocess(load_raw(subject=subject))
    epochs = make_epochs(raw)
    X, y   = extract_features(epochs)
    accs = []
    for seed in range(5):
        xa, xb, ya, yb = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=seed)
        c = Pipeline([("scaler", StandardScaler()),
                      ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))])
        c.fit(xa, ya)
        accs.append(c.score(xb, yb))
    return float(np.mean(accs))


# ── Sidebar controls ───────────────────────────────────────────────────────────

st.sidebar.header("Controls")
subject = st.sidebar.slider("Subject (PhysioNet ID)", min_value=1, max_value=20, value=1,
                            help="Each number is a different person's EEG recording.")
run = st.sidebar.button("▶ Run pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Compare across subjects")
n_subjects = st.sidebar.slider("How many subjects (1…N)", min_value=2, max_value=20, value=5,
                               help="Runs the classifier on subjects 1 through N and "
                                    "charts each one's accuracy.")
compare = st.sidebar.button("📊 Compare subjects", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "First run for a new subject downloads ~7 MB of EEG data, so it takes "
    "a moment. After that it's cached and instant."
)


# ── Main ────────────────────────────────────────────────────────────────────────

# Compare-across-subjects view (runs independently of the single-subject pipeline)
if compare:
    st.subheader(f"Accuracy across subjects 1–{n_subjects}")
    st.caption("Each bar is a different person. Notice how much it varies — some brains "
               "produce clear motor-imagery signals, others barely separable at all.")

    results = {}
    prog = st.progress(0.0, text="Starting…")
    for i, subj in enumerate(range(1, n_subjects + 1), start=1):
        prog.progress(i / n_subjects, text=f"Running subject {subj}…")
        try:
            results[subj] = quick_accuracy(subj)
        except Exception as e:
            st.warning(f"Subject {subj} failed: {e}")
    prog.empty()

    if results:
        subs = list(results.keys())
        accs = [results[s] for s in subs]
        mean_acc = np.mean(accs)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ["#55A868" if a >= 0.6 else "#C44E52" for a in accs]
        bars = ax.bar([f"S{s}" for s in subs], [a * 100 for a in accs], color=colors)
        ax.axhline(50, ls="--", color="gray", lw=1, label="Chance (50%)")
        ax.axhline(mean_acc * 100, ls=":", color="#4C72B0", lw=1.5,
                   label=f"Mean ({mean_acc*100:.0f}%)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Per-subject classification accuracy")
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

        best = max(results, key=results.get)
        worst = min(results, key=results.get)
        st.markdown(
            f"**What this shows:** Accuracy ranges from **{results[worst]*100:.0f}%** "
            f"(subject {worst}) to **{results[best]*100:.0f}%** (subject {best}), "
            f"averaging **{mean_acc*100:.0f}%**. This spread is expected: roughly 15–30% "
            "of people produce motor-imagery signals that are hard to decode — a known "
            "effect sometimes called *BCI illiteracy*. It's why reporting a single "
            "subject's score can be misleading, and why testing across people matters."
        )
    st.stop()

if not run:
    st.info("👈 Pick a subject in the sidebar and click **Run pipeline** to start. "
            "Or try **Compare subjects** to see how accuracy varies across people.")
    st.stop()

with st.spinner(f"Running the full pipeline for subject {subject}… "
                "(downloading data on first run)"):
    R = run_pipeline(subject)

st.success(f"Done! Analysed {R['n_epochs']} imagined movements "
           f"({R['n_left']} left, {R['n_right']} right).")

# ── Headline metrics ────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{R['test_acc']*100:.0f}%")
c2.metric("Cross-validated accuracy", f"{R['cv_mean']*100:.0f}%",
          help="A more honest score, averaged over 5 different splits.")
c3.metric("vs. random guessing", "50%", delta=f"{(R['cv_mean']-0.5)*100:+.0f} pts")

st.markdown("---")

# ── Plots ────────────────────────────────────────────────────────────────────────

left, right = st.columns(2)

# Raw EEG snippet
with left:
    st.subheader("Raw brain signal")
    st.caption("First 5 seconds across 8 sensors — the raw electrical activity.")
    fig, axes = plt.subplots(len(R["ch_names"]), 1, figsize=(7, 6), sharex=True)
    for i, (ax, ch) in enumerate(zip(axes, R["ch_names"])):
        ax.plot(R["raw_times"], R["raw_data"][i] * 1e6, lw=0.6, color="#4C72B0")
        ax.set_ylabel(ch, fontsize=7, rotation=0, labelpad=25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    st.pyplot(fig)

# Band power comparison
with right:
    st.subheader("Alpha & Beta power: Left vs Right")
    st.caption("How much energy each frequency band carries for each imagined hand.")
    bands  = {"Alpha (8–13 Hz)": (8, 13), "Beta (13–30 Hz)": (13, 30)}
    id2name = {v: k for k, v in R["event_id"].items()}
    groups = {"Left": [], "Right": []}
    side   = {"T1": "Left", "T2": "Right"}
    for epoch, ev in zip(R["epochs_data"], R["epoch_events"]):
        name = side.get(id2name.get(ev, ""), None)
        if name is None:
            continue
        for bname, (flo, fhi) in bands.items():
            freqs, psd = welch(epoch, R["sfreq"], nperseg=int(R["sfreq"]))
            idx = (freqs >= flo) & (freqs <= fhi)
            groups[name].append((bname, psd[:, idx].mean()))

    bnames = list(bands.keys())
    x = np.arange(len(bnames)); width = 0.35
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for i, (grp, color) in enumerate(zip(["Left", "Right"], ["#4C72B0", "#DD8452"])):
        means = [np.mean([p for b, p in groups[grp] if b == bn]) for bn in bnames]
        ax2.bar(x + i * width, means, width, label=f"{grp} hand", color=color)
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(bnames)
    ax2.set_ylabel("Mean band power (µV²/Hz)")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)

# Confusion matrix
st.markdown("---")
st.subheader("How well did the classifier do?")
st.caption("Rows = what was actually imagined. Columns = what the computer guessed. "
           "Dark squares on the diagonal mean correct guesses.")
cm_col, _ = st.columns([1, 1])
with cm_col:
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=R["cm"],
                                  display_labels=["Left Hand", "Right Hand"])
    disp.plot(ax=ax3, colorbar=False, cmap="Blues")
    ax3.set_title("Confusion Matrix — SVM")
    fig3.tight_layout()
    st.pyplot(fig3)

st.markdown("---")
with st.expander("ℹ️ What is this actually doing?"):
    st.markdown(
        "When you **imagine** moving your left or right hand, your brain produces "
        "slightly different electrical patterns — even without real movement. "
        "This app:\n\n"
        "1. **Loads** real EEG recordings of people imagining hand movements.\n"
        "2. **Cleans** the signal and slices it into labelled clips.\n"
        "3. **Measures** the energy in each frequency band for every sensor.\n"
        "4. **Trains** a machine-learning model (SVM) to tell left from right.\n"
        "5. **Tests** it on unseen clips and reports the accuracy.\n\n"
        "Above 50% means it's reading a real brain pattern, not guessing. This is the "
        "core idea behind brain-computer interfaces."
    )
