# Phase 3 Experiment Protocol

Version 1.0 (23 Oct 2026). This protocol was written and the physical regimes were defined **before** any Phase 3 classification result was examined. The only result seen beforehand was the timing run (seed 0, random split, 84.87 %), used solely to estimate runtime.

## 0. Status of the implementation

- The author's original pipeline, weights, metrics and environment are **unavailable**: no folder is attached and nothing was uploaded.
- All experiments use an **independent reimplementation** (`experiments/run_experiments.py`, TensorFlow 2.21.0 / Keras, scikit-learn 1.9.1, Python 3.13) of the configuration the manuscript publishes (Table 6, Table 5).
- Its results are labelled **"independent reimplementation"** everywhere. They are **not** reproductions of the original runs, and **must not** be substituted for them without the author's adoption and re-execution.

## 1. Data and provenance

| Item | Source | SHA-256 |
|---|---|---|
| classData.csv (7,861 × 10) | Publisher repository KingArthur000/Electrical-Fault-detection-and-classification, commit `4e094cf` (21 May 2021); same byte size (655,029 B) as the Kaggle file | `c8e7d1f2a6d4aefd74b5ed48d9ca4afdfdf2a5b676a5ed0935625e48972a36b2` |
| detect_dataset.xlsx (12,001 × 7) | same commit | `2eb28411bc79ba1b38dea729a1d6ed0bdbecddae0c6d0a71948d897883fc411a` |
| ANN_project.slx (parsed as XML; never executed) | same commit | `381dc4d052f7e295ee5cd9138673f2c037a98b6886da70f39638a07bbecdb3ff` |

## 2. Physical event timeline (established first)

**Sources:**
- Fault-block settings in the model: switch times, phases, R_on = R_g = 0.001 Ω.
- Solver: discrete, Ts = 50 µs, stop time 0.6 s.
- Measured currents in `detect_dataset.xlsx`.

The 12,001 rows are the 0.6 s timeline; every classData row maps to one timeline row (nearest-neighbour distance ≤ 4.6 × 10⁻⁸).

**Interruption criterion.** A faulted phase counts as interrupted at the first sample at or after the switch-off command from which its |I| stays below **200 A for the next 10 ms**.
- 200 A is about 2× the largest load-level boundary current observed (≈90–150 A) and far below fault peaks (≈880 A).
- Sensitivity at 150 A and 300 A is reported. The criterion is physical and was not tuned to any classification outcome.

**Regimes assigned to every labelled-fault row** (counts at 200 A):

| Regime | Definition | Rows | Label status |
|---|---|---|---|
| sustained | label start → switch-off command (includes the inception transient) | 4,960 | consistent |
| clearing_sametype | switch-off command → first faulted-pole interruption; every faulted pole still conducts | 207 | consistent (same topology) |
| partial_interruption | first → last faulted-pole interruption; the conducting topology differs from the label (LLG→A-G; LLL→A-B; LLLG→AB-G→A-G) | 288 | **fault-type inconsistent** |
| post_interruption | after the last faulted-pole interruption, still labelled fault | 41 | **mislabelled** (count threshold-sensitive: 1–18 per event) |

**How the original labels were assigned:**
- The publisher does not document the rule.
- Boundary currents range from 36 to 151 A, so it is not a single current threshold.
- Label starts lie 0.15 ms after inception (1.35 ms for LL).
- Label ends lie ≈0.3–0.5 ms after the last-pole interruption.

The rule is reported as **undocumented; boundaries characterised empirically**.

## 3. Datasets

| ID | Rows removed | Justification |
|---|---|---|
| E0 | none | Original labels (comparability) |
| E1 | partial_interruption + post_interruption (329) | **Demonstrably** type-inconsistent or mislabelled rows only |
| E2 | E1 + clearing_sametype (536) | Sensitivity: everything after the switch-off command removed (the Phase 2 diagnostic's variant, now with physically defined regimes) |

Removed rows that fall in a test partition are evaluated separately. Each prediction is scored against both the original label and the **physical topology** at that instant, where that topology exists in the label set.

## 4. Partitions

Blocks are the six contiguous class blocks of classData (row order = time order).

| Partition | Test set | Notes |
|---|---|---|
| random | stratified 20 %, seed-dependent | the manuscript's protocol |
| head | first 20 % of each block | contains the inception transient |
| mid | rows at 40–60 % of each block | interior of the sustained fault |
| tail | last 20 % of each block | the manuscript's blocked protocol; contains the clearing regimes |
| mid/tail + guard | as above, plus a 100-sample (¼-cycle) guard band each side of the test segment, excluded from training | adjacency control |

- Partitions are defined on the **original** blocks. E1/E2 rows are removed afterwards, so test positions are identical across datasets.
- Detection uses the 11 contiguous label runs of the detection file: random vs tail 20 % of each run.

## 5. Preprocessing and leakage controls

| Step | Primary (leakage-aware) | Comparison |
|---|---|---|
| Clip thresholds (1st/99th percentiles per raw channel) | fitted on the **training partition** | **global** (whole file), which reproduces Table 4 exactly |
| Scaler | fitted on the training partition (min-max) | robust (median/IQR), training-fitted; robust without clipping |
| Features | 24 per Table 5, computed after clipping | 23 features (V₀ removed) |
| Hyperparameters | fixed by the published configuration, so **no tuning on test data** | — |
| Decision rule | argmax (6-class); 0.5 threshold (detection) | — |
| Validation set | random 15 % of the **shuffled** training partition | unshuffled (Keras takes the last 15 %): tests the Phase 2 hypothesis S02 |

- **Duplicates.** No duplicate rows exist in classData (checked).
- **Temporal adjacency.** Random splits put every test row next to training rows. Blocked splits have at most two adjacent boundaries per block; the guard-band runs quantify this.

## 6. Model and training (published configuration)

| Setting | Value |
|---|---|
| Architecture | 24 → Dense 64 ReLU → Dropout 0.2 → Dense 32 ReLU → Dropout 0.2 → Dense 16 ReLU → 6 softmax (detection: 1 sigmoid) |
| Initialisation | He-normal |
| Optimiser | Adam, learning rate 1e-3 |
| Batch size | 32 |
| Epochs | ≤ 100 |
| Early stopping | patience 10, restore best |
| Learning-rate schedule | ReduceLROnPlateau, factor 0.5, patience 5 |
| Seeding | `tf.keras.utils.set_random_seed(seed)` |

- **Seeds:** 0–4, the same five for every condition, which enables paired comparisons.
- Five seeds are five **computational repetitions on one simulation**. They are **not** five independent fault events.

## 7. Grid (140 runs)

**Six-class** (5 seeds each):
- **E0, train-fitted clip, min-max:** {random, head, mid, tail}
- **E0, global clip, min-max:** {random, head, mid, tail}
- **E1:** {random, head, mid, tail}
- **E2:** {random, head, mid, tail}
- **Robust scaling (train clip; and no clip):** {random, mid, tail}
- **Guard band:** {mid, tail}
- **Unshuffled validation:** tail
- **23 features:** random

**Detection:** {random, tail}.

## 8. Outputs and analysis

**Per run** (JSON, one file each):
- train and test counts;
- accuracy, merged LLL+LLLG accuracy;
- confusion matrix;
- per-class precision, recall, F1 and support;
- epochs, best validation loss;
- test-regime counts and accuracy by regime;
- predictions on excluded test rows, with original and physical labels.

**Per condition:**
- every seed's value, mean, sample s.d.;
- a 95 % t-interval across seeds (n = 5; describes computational variability only);
- class-wise mean ± s.d.;
- a summed confusion matrix.

**Paired comparisons** (same seed), each as mean difference, s.d., 95 % t-interval and a sign count:
- tail − random;
- tail − mid;
- E1 − E0 within a partition;
- global − train clip.

**Interpretation rules (fixed in advance):**
- A partition effect attributed to **temporal adjacency** requires random > mid ≈ guard-mid.
- An effect attributed to **label/regime contamination** requires the tail deficit to shrink materially from E0 to E1 (and E2), and errors on excluded rows to follow their physical topology.
- **Memorisation** is not claimed unless adjacency controls show it.
- **Generalisation to unseen fault events cannot be tested:** each class is a single event.
- **Class-specific claims** (e.g. LLLG recall) are reported only with their across-seed range.

## 9. What this protocol cannot establish

- Behaviour of the author's original pipeline.
- Performance on independent fault events, other fault resistances, locations, inception angles or phase pairs.
- Any relay-timing or field property.

## Amendments (after training; descriptive corrections only — no definition, criterion or analysis rule changed)

| Date | Section | Change | Reason |
|---|---|---|---|
| 23 Oct 2026 | §2 | "Label starts lie 0.15 ms after inception" → **0.15–0.20 ms** (LL: 1.35 ms) | Recomputed by `data_facts.py`; the LG label starts 0.20 ms after inception |
| 23 Oct 2026 | §2 | "Label ends lie ≈0.3–0.5 ms after the last-pole interruption" → **0.20–0.50 ms** | Recomputed by `data_facts.py` (LL: 0.20 ms) |
| 23 Oct 2026 | §2 | Post-interruption sensitivity "3–18 per event" → **1–18 per event** (150–300 A) | Recomputed; the 150-A minimum is 1 (LL) |
| 23 Oct 2026 | §2 | Added: partial-interruption counts change by at most **13** rows per event between 150 A and 300 A (LLL) | Recomputed |

---

# Addendum B — Phase 3b extension analyses (pre-registered 23 Oct 2026, before any Phase 3b run)

Purpose: test whether the Phase 3 findings are specific to the MLP, whether high LLL/LLLG accuracy depends on the constant residual offset, how much independent information the rows carry, and where detection errors fall. Data, regimes, partitions (random/middle/tail; E0/E1), seeds (0–4) and train-only fitting are exactly as in §§2–5. All runs are independent implementations; none reproduces the original pipeline.

## B1. Model-agnostic partition and contamination test
- Models (scikit-learn 1.9.1, defaults unless stated): logistic regression (multinomial, max_iter 2000), k-nearest neighbours (k = 5), decision tree, random forest (200 trees), histogram gradient boosting; each with `random_state = seed` where applicable.
- Features: the 24 of Table 5 computed on unclipped channels (trees use raw features; LR and kNN use a standard scaler fitted on training rows only).
- Grid: {random, mid, tail} × {E0, E1} × 5 seeds. Metrics: six-class accuracy, merged LLL/LLLG accuracy, LLL/LLLG pair-only accuracy, non-pair errors.
- Pre-stated reading: if every model shows tail < mid ≈ random under E0 and the tail deficit disappears under E1, the effect is a property of the data, not of the MLP.

## B2. Shortcut test (offset and noise sensitivity at test time)
- Train on the random split (E0) as in B1 (trees, kNN, LR) and with the Keras MLP using robust scaling without clipping (the configuration that resolved the pair in Phase 3); 5 seeds.
- Test-time perturbations applied to the three phase currents of the **test** rows only:
  (a) common-mode offset δ added equally to Ia, Ib, Ic (shifts I0 by δ, leaves line currents' differences unchanged): δ ∈ {−1.0, −0.7, −0.35, 0, +0.35, +0.7, +1.0} A;
  (b) independent zero-mean Gaussian noise per phase and sample, σ ∈ {0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10} A.
  Voltages unchanged. Features recomputed from perturbed channels; the training-fitted scaler is reused.
- Metrics: LLL/LLLG pair-only accuracy; accuracy on the four other classes (non-pair accuracy); merged accuracy.
- Pre-stated reading: if pair accuracy of models that resolve the pair collapses to ≈ chance for |δ| ≈ 0.7 A or σ ≲ 1 A while non-pair accuracy is unchanged, the pair decision rests on a sub-ampere quantity (the constant offset) rather than on fault physics. No claim is made about specific instrument-transformer accuracy classes.

## B3. Effective sample size (learning curve)
- Random split (E0) test set fixed per seed; training uses a stratified random fraction f ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0} of the training partition. Models: random forest and Keras MLP (min–max, train-fitted clip, published configuration). 5 seeds.
- Pre-stated reading: a curve that saturates at small f indicates that rows are highly redundant samples of a few events.

## B4. Where detection errors fall
- Keras detection model (published configuration, train-fitted clip, min–max), random split, 5 seeds; per-row predictions saved.
- For each misclassified test row: time to the nearest label transition (ms) and regime (sustained, clearing, partial, post, healthy).
- Pre-stated reading: if errors concentrate within ~1 ms of label boundaries, residual detection error reflects the undocumented labelling rule rather than missed faults.

## B5. Not changed
Regime definitions, partitions, seeds, exclusion criteria, and all Phase 3 results remain as reported.

# Addendum C — Phase 3c analyses (pre-registered 23 Oct 2026, after Addendum B results and before any Phase 3c run)

Purpose: three checks that bear directly on claims made from Addendum B. Same data, partitions, seeds (0–4) and train-only fitting as §§2–5. Independent implementation; none reproduces the original pipeline.

## C1. Which features carry the three-phase decision (random forest)
- Random forest as in B1 (200 trees, raw unclipped features), random split, E0, 5 seeds.
- Permutation importance on the test set (10 permutations per feature, `random_state = seed`), scored two ways: (a) six-class accuracy on all test rows; (b) pair-only accuracy on LLL/LLLG test rows.
- Pre-stated reading: if the zero-sequence-current feature (and only features that change under a common-mode current shift) carries most of the pair importance, the pair decision rests on the residual offset, consistent with B2. Correlated features can share importance; this is reported, not corrected.

## C2. Sensitivity of the label-contamination result to the interruption threshold
- Regimes recomputed with the 150-A and 300-A criteria (10-ms hold unchanged), E1 rebuilt for each.
- Random forest (B1 configuration) and the network (published configuration, train clip, min–max), tail hold-out, E0 vs E1, 5 seeds.
- Pre-stated reading: if the tail E1 − E0 gain changes by less than 2 percentage points across 150/200/300 A for both models, the conclusion does not depend on the threshold.

## C3. Relabelling by circuit state instead of exclusion (variant E3)
- E3: partial-interruption rows whose conducting topology is a class are relabelled to that class (LLG → A–G; LLLG → AB–G, then A–G); post-interruption rows are relabelled no-fault; LLL partial-interruption rows (A–B conducting, not a class) are removed. 200-A criterion.
- Random forest and the network as in C2; random and tail hold-outs; 5 seeds.
- Pre-stated reading: if E3 tail accuracy matches E1 tail accuracy within 2 points, defining labels from the circuit state removes the hold-out deficit without discarding the interruption interval.

## C4. Not changed
Everything in Phase 3 and Addendum B stands as reported.

# Addendum D — error composition (registered 23 Oct 2026, after Addendum C, before its run)

## D1. Where each model's tail errors lie
- The five B1 models (same configuration, raw features) are refitted on the tail hold-out, E0, seeds 0–4, and every test error is assigned to its regime (sustained, clearing same-type, partial interruption, post-interruption, healthy). For the network, the per-regime accuracies already stored in the Phase 3 summary are converted to error counts.
- Descriptive only; no pre-stated threshold. The analysis tests whether the share of tail errors on the 329 label-inconsistent rows is similar across families.
