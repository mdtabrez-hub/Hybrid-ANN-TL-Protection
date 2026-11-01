"""Single place that sets where the scripts read data and write results.

ROOT  : repository root (results/, figures/ are written here).
DATA  : clone of KingArthur000/Electrical-Fault-detection-and-classification at commit 4e094cf
        (default: <ROOT>/data/pubrepo; override with the FAULT_DATA_DIR environment variable).
WORK  : scratch directory for intermediate pickles (default: <ROOT>/work).

Importing this module also changes the working directory to ROOT, because several scripts
use paths relative to the repository root (e.g. 'results/...').
"""
import os

ROOT = os.path.abspath(os.environ.get('REPRO_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
DATA = os.path.abspath(os.environ.get('FAULT_DATA_DIR', os.path.join(ROOT, 'data', 'pubrepo'))) + '/'
WORK = os.path.abspath(os.environ.get('REPRO_WORK_DIR', os.path.join(ROOT, 'work')))
for _d in (WORK, os.path.join(ROOT, 'results'), os.path.join(ROOT, 'figures')):
    os.makedirs(_d, exist_ok=True)
if not os.path.exists(DATA + 'classData.csv'):
    raise SystemExit(f'Dataset not found in {DATA}. Run scripts/fetch_data.sh or set FAULT_DATA_DIR.')
os.chdir(ROOT)
