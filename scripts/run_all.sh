#!/usr/bin/env bash
# Run the full pipeline in the order the results depend on each other.
# Training steps are resumable: runs whose JSON already exists in results/ are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo ">> python code/$1"; python "code/$1"; }
run timeline.py; run checks.py; run checks2.py
run timeline_physics.py; run regimes.py; run data_facts.py
run make_data_figures.py; run plot_events.py; run make_topology_figure.py
run run_experiments.py; run aggregate.py; run derived.py; run results_tables.py
run run_phase3b.py; run aggregate_phase3b.py; run b4_localise.py
run run_phase3c.py; run diag_e3.py; run aggregate_phase3c.py
run run_phase3d.py
echo "Done. Outputs are in results/ and figures/."
