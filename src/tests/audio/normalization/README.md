# Volume Normalization Test

This folder is a local test harness for loudness normalization before applying the process to the full dataset rebuild.

## Structure

- `input/` — drop test clips here (mixed loudness: some already good, some too quiet)
- `output/` — normalized outputs are written here as `*_norm.wav`
- `normalize_volume.py` — local script for EBU R128 normalization
- `mic_pop_audit.py` — local script to flag likely clip-start mic pops
- `normalization_report.csv` — per-file results (generated)
- `normalization_summary.json` — aggregate results (generated)

Additional mic-pop outputs:

- `mic_pop_report.csv` — per-file detector metrics and flag status
- `mic_pop_summary.json` — aggregate flagged percentage and recommendation
- `mic_pop_flagged.txt` — plain list of flagged filenames

## Recommendation (handover)

Use EBU R128 integrated loudness normalization with:

- target: `-23 LUFS`
- skip tolerance: `+/-1 LU`
- post-gain clipping protection: `np.clip(audio, -1.0, 1.0)`

This aligns with the Phase A rebuild plan in `.docs/handover.md` and keeps behavior deterministic for the full pass.

## Run

From repo root:

```bash
python src/tests/audio/normalization/normalize_volume.py
```

Run mic-pop audit:

```bash
python src/tests/audio/normalization/mic_pop_audit.py
```

Optional tuning:

```bash
python src/tests/audio/normalization/normalize_volume.py --target-lufs -23 --tolerance-lu 1
```

## Notes

- If integrated loudness cannot be measured (very short/silent/broken files), the clip is passed through and marked failed in the report.
- The script preserves the original sample rate of each input file for this local test stage.
