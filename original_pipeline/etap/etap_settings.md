# ETAP 19 coordination model and test plan

**STATUS: PENDING EXTERNAL EXECUTION.** No ETAP result appears in the
manuscript. This file defines the model, the study matrix and the reporting
rules required to produce one. ETAP 19.0.1+ with the Star Protection &
Coordination module is required.

## 1. Blocking prerequisite

The supplied CSVs contain **no network parameters** — no base voltage, no line
impedance, no source impedance, no fault resistance and no fault location. The
Blueprint's network description (four 11 kV sources, two transformers, a
220 kV / 100 km line) is a *proposal*, not a record of what generated the data.

An ETAP model built on assumed parameters would produce trip times that cannot
legitimately be compared with the ANN. Therefore:

> `[USER INPUT REQUIRED]` Supply the original Simulink model (`.slx`) or its
> parameter sheet: source MVA and X/R, transformer ratings and impedances, line
> length and sequence impedances, CT and VT ratios, and the fault resistance and
> location used for each of the six runs.

Until these are supplied, Sections 9 and 10 of the manuscript remain pending
and **no comparative improvement figure may be stated**.

## 2. One-line diagram

| Element | Specification | Source |
|---|---|---|
| Generators | 4 × 11 kV, 50 Hz | Blueprint 4.1 — unconfirmed |
| Transformers | 2 × step-up / step-down | Blueprint 4.1 — unconfirmed |
| Transmission line | 100 km, 220 kV | Blueprint 4.1 — unconfirmed |
| Loads | representative distribution | `[USER INPUT REQUIRED]` |

## 3. Protection functions

| Device | ANSI | Setting parameters | Status |
|---|---|---|---|
| Distance | 21 | Zone 1 = 80 % of line, Zone 2 = 120 %, Zone 3 = backup | `[USER INPUT REQUIRED]` |
| Overcurrent | 50/51 | pickup, curve (IEC/IEEE), TMS | `[USER INPUT REQUIRED]` |
| Ground fault | 51G | residual pickup, curve, TMS | `[USER INPUT REQUIRED]` |

## 4. Study matrix (Blueprint 4.4)

- Fault resistance R_f ∈ {0, 5, 20, 50} Ω
- Fault location ∈ {10, 25, 50, 75, 90} % of line length
- Source strength ∈ {strong grid, weak grid}
- Fault type ∈ {LG, LL, LLG, LLL, LLLG}

= 200 short-circuit cases. Note that the **supplied dataset covers exactly one
cell of this matrix per fault type** — a single fixed configuration — which is
the finding reported in Section 6.5 of the manuscript. Regenerating the
training data across this same matrix is the primary recommendation.

## 5. Procedure

1. Build the one-line diagram; run Load Flow and confirm convergence.
2. Run Short-Circuit Analysis (IEC 60909 or ANSI; state which).
3. Enter relay settings; open Star → Protection & Coordination.
4. Run Auto-Evaluation; resolve every reported miscoordination.
5. Export TCC curves and the relay operating time for each case.
6. Verify the coordination time margin of Eq. (19) for every primary/backup
   pair: CTM = t_backup − t_primary − Δt_security ≥ 0, with Δt_security
   typically 0.2–0.4 s (state the value used).

## 6. Comparison rules

The ANN and the ETAP relays must be compared **like for like**:

- ETAP relay operating time excludes circuit-breaker time; either add breaker
  time to both or exclude it from both, and say which.
- The ANN figure must be the measured Proteus trip latency (Proteus test P5),
  not the host inference time.
- The serial-link latency (test P4) must be subtracted or disclosed.
- Report per fault type, not as a single headline percentage.

An improvement figure computed from an ETAP time and a *host* Python inference
time is not a valid comparison and must not be reported.

## 7. Deliverables

- `.etap` project file
- TCC plots per relay pair (Fig. 13)
- Trip-time table per case (Fig. 14)
- Sensitivity results across the study matrix (Fig. 15)
