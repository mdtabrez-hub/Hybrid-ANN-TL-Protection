# Original pipeline (earlier, Elsevier-format version of the study)

This folder holds the code behind the **earlier version** of the manuscript, *"Executing the verification chain: what a public three-phase fault dataset can and cannot support for neural transmission-line protection"* (September 2026). That version covered the full Python → MATLAB/Octave → C → Proteus → ETAP chain.

The current manuscript (IEEE TPWRD version) does **not** depend on anything in this folder. Its results come from the independent implementation in `../code/`. This folder is kept so the numbers in the earlier version can also be reproduced, and because the current manuscript compares against them (`../results/original_reported_values.json`).

## Contents

| Path | What | Status in the earlier version |
|---|---|---|
| `src/feature_engineering.py` | The 24 row-wise features | executed |
| `src/train_ann.py`, `src/train_ann.ipynb` | Data audit, multi-label / detection / six-class / SMOTE networks (Keras), exports for later stages | executed |
| `src/audit.py`, `src/label_physics.py`, `src/separability.py`, `src/probe_llll.py` | Label-vector audit and the LLL/LLLG separability analysis | executed |
| `src/scaling_ablation.py`, `src/clipping_ablation.py` | Scaler and clipping ablations | executed |
| `src/blocked_split.py` | Random versus blocked partition (temporal-leakage check) | executed |
| `src/ann_predict.py`, `src/validate_generate_c.py` | NumPy port of the MATLAB stage; cross-implementation check; int8 quantisation | executed |
| `src/make_figures.py`, `src/plotstyle.py` | Figures of the earlier version | executed |
| `matlab/ann_predict.m`, `matlab/validate_octave.m`, `matlab/emit_c.m`, `matlab/+coder/load.m` | Forward pass, Octave validation and C emitter (runs under GNU Octave; `+coder/load.m` is a shim) | executed in Octave |
| `matlab/validate_generate_c.m` | MATLAB Coder script | **not executed** (no MATLAB licence) |
| `phase2/test_host.c`, `phase2/main_avr.c`, `phase2/bench.c` | Host test harness, ATmega328P entry point, timing benchmark for the emitted C | executed (host gcc, avr-gcc) |
| `proteus/transmission_line_protection.ino`, `proteus/proteus_settings.md` | Arduino firmware scaffold and hardware-in-the-loop test plan | **not executed** (out of scope in the final earlier version) |
| `etap/network.py`, `etap/calibrate_network.py`, `etap/relays.py`, `etap/run_study.py`, `etap/make_etap_figures.py` | Relay-coordination study on a network calibrated to the dataset's fault currents, solved with **pandapower** (IEC 60909) and an independent sequence-network solver | executed — this is an ETAP-equivalent study, **not run in ETAP** |
| `etap/etap_settings.md` | The original ETAP 19 study plan, written before the pandapower study replaced it | plan only |
| `data/prepare_data.py` | Puts the two CSVs in `data/` (see below) | — |

## Data

The scripts read `classData.csv` and `detect_dataset.csv` from `data/` (or from the folder named in the `FAULT_DATA_DIR` environment variable). They are not redistributed. Either:

- download both CSVs from Kaggle, "Electrical Fault detection and classification" (E. Sathya Prakash, version 2, CC0), into `data/`; or
- from the repository root run `bash scripts/fetch_data.sh`, then `python original_pipeline/data/prepare_data.py`. This copies `classData.csv` from the GitHub copy and writes `detect_dataset.csv` from its XLSX.

## How to run

Run from this folder (`original_pipeline/`); outputs go to `out/`, `figs/` and `etap/results/`.

```bash
pip install -r requirements.txt
python src/train_ann.py              # ~3 min on 2 CPU threads; writes out/ (weights, test set, metrics)
python src/audit.py; python src/label_physics.py; python src/separability.py; python src/probe_llll.py
python src/scaling_ablation.py; python src/clipping_ablation.py     # ~5 min each
python src/blocked_split.py          # ~10 min
python src/validate_generate_c.py    # NumPy cross-check and int8 quantisation
# Octave stage (needs GNU Octave):  cd matlab && octave validate_octave.m && octave emit_c.m
# then compile phase2/*.c with the emitted out/phase2/ann_predict*.c (host gcc; avr-gcc for the footprint)
python etap/calibrate_network.py; python etap/run_study.py; python etap/make_etap_figures.py
python src/make_figures.py           # needs the Octave-stage outputs in out/phase2/
```

## Check on 1 November 2026

The Python and pandapower stages were re-run from this folder, using the GitHub copy of the data via `data/prepare_data.py`, Python 3.13 and the versions in `requirements.txt`. Results against the earlier version:

| Quantity | Earlier version | Re-run |
|---|---|---|
| Binary detection accuracy (random split) | 99.79 % | 99.79 % |
| Multi-label exact match | 84.49 % | 84.49 % |
| Six-class accuracy | 85.57 % | 85.57 % |
| NumPy vs Keras discrepancy | 0.000000 % | 0.000000 % |
| int8 quantisation cost | −0.25 pp | −0.25 pp |
| Scaler and clipping ablations (9 variants) | as reported | identical to the last digit |
| Blocked split: six-class / merged / multi-label | 74.00 % / 82.45 % / 74.95 % | identical |
| Blocked split: detection accuracy | 99.63 % | 99.71 % |
| Coordination study: unfitted-fault error, mean (worst) | 2.38 % (4.53 %) | 2.38 % (4.53 %) |
| Coordination study: cases cleared | 200 of 200 | 200 of 200 |

The only difference is blocked-split detection (99.71 % against 99.63 %), most likely because the re-run read the detection data from the GitHub XLSX rather than the Kaggle CSV. The Octave, C, AVR and Proteus stages were not re-run here.

## Changes from the version used in September 2026

- The data folder is configurable (`FAULT_DATA_DIR`, default `data/`) instead of a fixed path.
- `etap/make_etap_figures.py` creates `figs/` if it does not exist.

No other line was changed.
