# Limitations and known issues

1. **One event per class.** The dataset is a single 0.6 s simulation with one bolted mid-line fault per class. No partition of it can test generalisation to unseen fault events, resistances, locations, inception angles or phase pairs.
2. **Independent implementation.** The original pipeline was unavailable, so results are not reproductions of the original study.
3. **Labelling rule undocumented.** Label boundaries are characterised empirically (Protocol §2). The post-interruption regime count depends on the interruption threshold (1–18 rows per event between 150 A and 300 A).
4. **Simulink model values unreconciled.** A steady-state estimate from the stored source and load values does not reproduce the CSV current levels. The model file may have been edited after the data were generated, so only its structure (topology, fault settings, solver step) is relied on.
5. **Seeds are not events.** t-intervals over five seeds describe training variability only.
6. **Line-to-line voltages.** The zero-sequence-voltage feature is identically zero before clipping and a clipping artefact after it. It is retained only for comparability with the original feature set; its removal is tested separately.
7. **Platform nondeterminism.** oneDNN and threading may change results slightly across machines.
8. **Licensing.** The GitHub copy of the data and model is unlicensed; the Kaggle copy of the data is CC0. Data are not redistributed.
9. **Paths.** All paths are set in `code/_paths.py` (repository root, dataset location via `FAULT_DATA_DIR`, scratch folder). The pipeline was checked from a clean copy of this repository; see the README for what was regenerated.
10. **Further classifiers untuned.** The five Addendum B models use library defaults. Absolute accuracies could change with tuning; the label-inconsistent rows they misclassify would not.
11. **E3 relabelling.** The pre-stated E3 criterion (agreement with E1 within 2 points) was not met. The explanation offered (relabelled interruption rows absent from training under the tail hold-out) comes from a post-hoc check (`diag_e3.py`), labelled as such.
12. **Perturbation probes.** They are diagnostic and are not a model of instrument-transformer or acquisition error.
13. **Literature values.** Values from Anwar et al. were read in the full text; those from Shamim et al. come from the publisher's abstract page only.
14. **Phase 3b–3d run time.** Addenda B–D add about 1 h of CPU time on 2 threads; all scripts are resumable.
