# Auditing a public transmission-line fault dataset — reproducibility code

Code, protocol and results for the manuscript

> **Auditing a Public Transmission-Line Fault Dataset: Single-Event Simulation, Fault-Clearing Labels, and Their Effect on Classifier Evaluation**
> Md Tabrez, Department of Electrical and Electronics Engineering, Motihari College of Engineering, Motihari, Bihar, India.
> Manuscript prepared for *IEEE Transactions on Power Delivery* (under preparation; not yet published).

This repository reproduces:

- every dataset-level fact cited in the manuscript (event timeline, fault-clearing regimes, label boundaries, zero-sequence statistics);
- every classifier result in the manuscript (140-run neural-network grid, five further model families, perturbation probes, learning curve, permutation importance, threshold and relabelling checks, error composition);
- all manuscript figures.

The classifiers are an **independent implementation** of the configuration reported in the earlier version of the study (24 features; 64-32-16 ReLU MLP; dropout 0.2; Adam 1e-3; batch 32; ≤100 epochs; early stopping 10; learning-rate halving 5).

## Repository layout

| Path | What |
|---|---|
| `code/_paths.py` | Sets the repository root, the dataset location and a scratch directory. Edit nothing else to relocate. |
| `code/parse_slx.py`, `code/topo.py` | Read the Simulink model as XML (block parameters; electrical connectivity). The model is never executed. |
| `code/timeline.py`, `code/checks.py`, `code/checks2.py` | Map each classification row onto the 0.6 s timeline; dataset statistics |
| `code/timeline_physics.py`, `code/regimes.py` | Per-phase interruption times; assign each row to a fault-clearing regime (`results/detect_row_regimes.csv`, `results/event_timeline.json`) |
| `code/data_facts.py` | Every dataset number cited in the manuscript (`results/data_facts.json`) |
| `code/make_data_figures.py`, `code/plot_events.py`, `code/make_topology_figure.py` | Figs. 1, 2 and 4 and the event-window audit plot |
| `code/run_experiments.py` | The 140-run neural-network grid (one JSON per run; resumable) |
| `code/aggregate.py`, `code/derived.py`, `code/results_tables.py` | Summaries, paired comparisons, excluded-row evaluation, derived quantities, results tables; Figs. 3 and 5 |
| `code/run_phase3b.py`, `code/aggregate_phase3b.py`, `code/b4_localise.py` | Protocol Addendum B: five further model families (B1), perturbation probes (B2), learning curve (B3), detection-error localisation (B4); Figs. 8–9 |
| `code/run_phase3c.py`, `code/aggregate_phase3c.py`, `code/diag_e3.py` | Protocol Addendum C: permutation importance (C1), interruption-threshold sensitivity (C2), relabelling by circuit state (C3); Fig. 7. `diag_e3.py` is a post-hoc check, labelled as such. |
| `code/run_phase3d.py` | Protocol Addendum D: tail-error composition by regime for every model family; Fig. 6 |
| `docs/PHASE3_EXPERIMENT_PROTOCOL.md` | The analysis protocol, fixed before training, with Addenda B–D |
| `results/` | All outputs used in the manuscript: per-run JSON (`runs/`, `phase3b/`, `phase3c/`), run tables, summaries, `data_facts.json`, `derived.json`, `RESULTS_TABLES.md`. `original_reported_values.json` holds the earlier version's reported numbers for comparison; `literature_values.json` holds values quoted from other studies. |
| `figures/` | Manuscript figures (see the mapping below) |
| `data/checksums.sha256` | SHA-256 of the three input files. The data are not redistributed (see below). |
| `scripts/fetch_data.sh`, `scripts/run_all.sh` | Fetch and verify the dataset; run the whole pipeline |
| `environment.txt`, `requirements.txt` | Library versions used |
| `LIMITATIONS.md` | Known issues |

Figure file → manuscript figure: `fig0_topology` → 1, `fig1_timeline_col` → 2, `fig3_confusion` → 3, `fig2_zero_sequence` → 4, `fig4_partitions` → 5, `fig9_error_composition` → 6, `fig7_importance` → 7, `fig5_shortcut` → 8, `fig6_learning_curve` → 9.

## Data (third-party; not redistributed)

| File | Source | Licence |
|---|---|---|
| `classData.csv`, `detect_dataset.csv` | Kaggle, "Electrical Fault detection and classification", E. Sathya Prakash, version 2 (22 May 2021) | CC0: Public Domain |
| `classData.csv`, `detect_dataset.xlsx`, `ANN_project.slx` | GitHub [`KingArthur000/Electrical-Fault-detection-and-classification`](https://github.com/KingArthur000/Electrical-Fault-detection-and-classification), commit `4e094cf` (21 May 2021) | No licence file |

The analysis uses the GitHub copies. Because that repository carries no licence, this repository does not copy the files; `scripts/fetch_data.sh` clones them into `data/pubrepo/` and checks them against `data/checksums.sha256`. The Kaggle detection file is a CSV with two extra empty columns, so the code reads the XLSX.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/fetch_data.sh          # clones the dataset into data/pubrepo and verifies SHA-256
bash scripts/run_all.sh             # full pipeline; about 2 h on 2 CPU threads
```

To use a dataset clone elsewhere, set `FAULT_DATA_DIR=/path/to/clone`. All scripts can be run from any directory; they write to `results/`, `figures/` and a scratch folder `work/`.

### Execution order

```bash
python code/timeline.py && python code/checks.py && python code/checks2.py
python code/timeline_physics.py
python code/regimes.py              # clearing regimes (200 A / 10 ms criterion)
python code/data_facts.py
python code/make_data_figures.py    # Figs. 2 and 4
python code/plot_events.py
python code/make_topology_figure.py # Fig. 1
python code/run_experiments.py      # 140 runs, ~25 s each on 2 CPU threads (~1 h); resumable
python code/aggregate.py            # Figs. 3 and 5
python code/derived.py && python code/results_tables.py
python code/run_phase3b.py          # Addendum B (~45 min); resumable
python code/aggregate_phase3b.py && python code/b4_localise.py   # b4_localise must follow aggregate
python code/run_phase3c.py          # Addendum C (~15 min); resumable
python code/diag_e3.py && python code/aggregate_phase3c.py       # diag_e3 before aggregate
python code/run_phase3d.py          # Addendum D
```

The run scripts skip any run whose JSON already exists in `results/`. To retrain from scratch, delete `results/runs/`, `results/phase3b/` and `results/phase3c/` first; to rebuild only the summaries and figures, keep them.

## Environment and determinism

Python 3.13, TensorFlow 2.21.0 (Keras 3.15), scikit-learn 1.9.1, NumPy 2.5, pandas 3.0, SciPy 1.18, matplotlib 3.11, openpyxl 3.1, CPU only, 2 threads (`environment.txt`). Seeds are fixed with `tf.keras.utils.set_random_seed`, and oneDNN optimisations are disabled.

Checked from a clean copy of this repository on 1 Nov 2026 with the same library versions: the dataset facts, regime assignment and event timeline were regenerated identically; a retrained grid run matched its stored result to every reported digit; and all summaries (`summary_by_condition.csv`, `paired_comparisons.csv`, `derived.json`, `phase3b_summary.json`, `phase3c_summary.json`) were regenerated identically from the stored runs. On other hardware or library versions, expect agreement within seed-to-seed variability rather than to the last digit.

## What the results are, and are not

- Five seeds are repeated computations on **one** simulation. They are not independent fault events, and their intervals describe computational variability only.
- The dataset holds **one fault event per class**, so no result here estimates performance on unseen faults.
- See `LIMITATIONS.md` for the full list.

## Citation

If you use this code, please cite the manuscript (details will be added on publication) and this repository (`CITATION.cff`).

## Licence

Code and documentation: MIT (see `LICENSE`). The dataset is third-party and is not covered by this licence.
