# Artifact Detection — Attempted Approach & Decision to Drop

## Problem

During manual review of the curated sample clips, a subset of recordings was found to contain audible audio artifacts — specifically:

- **Impulsive noise bursts**: crackling, popping, or firecracker-like sounds mid-clip, likely caused by microphone handling noise, electrical interference, or sudden mic feedback.
- **Mic drop/switch transients**: sharp onset spikes at the start or end of a recording where the microphone was physically disturbed.

These artifacts were identified as a data quality concern because they could introduce noise patterns into TTS model training that do not represent the target speaker's voice.

A concrete example was identified: `sna_sna_1296_raw.wav`, a 30-second clip where clean speech transitions into crackling artifact noise from approximately 15.5 seconds onward, clearly visible as broadband vertical bursts in the Mel spectrogram.

---

## Attempted Detection Method

A signal-processing detection script was developed (`src/tests/artifact_check/detect.py`) using two independent acoustic signals:

### 1. Per-window Kurtosis (Impulse Detection)
- Audio was segmented into short windows (~20ms).
- Excess kurtosis (Fisher definition) was computed per window.
- High kurtosis indicates impulsive, heavy-tailed signal distributions — characteristic of clicks and pops.
- Threshold: kurtosis > 25 (raised iteratively from 10).
- Minimum spike count: 3 distinct events required to flag.

### 2. High-Frequency Energy Ratio (Broadband Burst Detection)
- STFT was computed on 16kHz-resampled audio.
- Per-frame ratio of energy above 5kHz to total energy was measured.
- Real artifacts tend to spread energy broadband (including high frequencies); clean speech at 16kHz stays mostly below 5kHz.
- Threshold: HF ratio > 0.55.

### Rejection Logic
A clip was rejected only when **both** signals agreed **and** their spike events were temporally co-located (within 2 seconds of each other). This AND + correlation requirement was intended to reduce false positives from natural speech features.

Additional mitigations applied:
- 0.5-second grace period at clip start for both detectors (to ignore mic-open transients and speech onset plosives).
- All analysis performed on 16kHz-resampled audio regardless of source sample rate, normalising the HF profile across clips.

---

## Why It Was Dropped

Despite iterative threshold tuning, the method produced an unacceptably high false positive rate (~93% of clips flagged in a 100-clip random sample). Investigation identified the following fundamental limitations:

1. **Natural speech has high kurtosis.** Plosive consonants (p, t, k, b, d, g), glottal stops, and fast consonant clusters produce frame-level kurtosis values that overlap significantly with impulsive artifact kurtosis. This is especially pronounced in Shona, which has a rich consonantal inventory.

2. **HF energy is distributed throughout the dataset.** The Shona recordings span many different recording conditions and microphone types. A fixed HF ratio threshold cannot distinguish between a clip recorded on a high-bandwidth microphone (which naturally captures more HF content) and a clip with broadband noise artifacts.

3. **No clean baseline.** Without a labelled set of known-clean and known-artifact clips large enough to calibrate against, there is no principled way to set thresholds that generalise across the full dataset.

4. **The problem is sparse.** Artifacts of this type appear to affect a small minority of the dataset. Aggressive filtering to catch them risks discarding substantially more clean data than artifact data — a net negative for dataset size and coverage.

---

## Alternative Considered

More robust approaches exist — such as DNSMOS (Microsoft's deep noise suppression MOS predictor) or NISQA (Neural Image and Speech Quality Assessment) — which use pre-trained models to predict perceptual audio quality scores rather than hand-crafted signal features. These would likely perform better on this task.

However, given the scope of this project and the fact that the artifact rate appears low enough not to materially impact downstream model training, the decision was made to accept the dataset as-is after VAD-based curation and defer quality-model-based filtering to future work.

---

## Outcome

- The `src/tests/artifact_check/` directory and `detect.py` script are retained in the repository as a record of the attempt.
- No artifact-based filtering is applied in the production pipeline.
- The `sna_sna_1296_raw.wav` clip (identified artifact) and `sna_sna_2586_refined.wav` clip (confirmed clean) remain in `src/tests/artifact_check/input/` as reference benchmarks for any future attempt.
