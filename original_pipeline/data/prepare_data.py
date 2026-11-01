"""Create classData.csv and detect_dataset.csv in this folder for the original pipeline.

Preferred source: the Kaggle dataset "Electrical Fault detection and classification"
(E. Sathya Prakash, v2, CC0) -- download both CSVs and place them here; then this
script is not needed.

Fallback (used for the check described in ../README.md): the GitHub copy
KingArthur000/Electrical-Fault-detection-and-classification at commit 4e094cf,
which ships classData.csv (byte-identical size to Kaggle) and detect_dataset.xlsx.
The XLSX is written out as CSV with the two empty trailing columns the Kaggle CSV has.

Usage:  python data/prepare_data.py [path/to/clone]
        (default clone path: ../data/pubrepo, i.e. the one scripts/fetch_data.sh creates)
"""
import os, sys, shutil
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'data', 'pubrepo')
if not os.path.exists(os.path.join(src, 'classData.csv')):
    sys.exit(f'No dataset clone at {src}. Run scripts/fetch_data.sh from the repository root first.')
shutil.copy(os.path.join(src, 'classData.csv'), os.path.join(HERE, 'classData.csv'))
d = pd.read_excel(os.path.join(src, 'detect_dataset.xlsx'))
d['Unnamed: 7'] = None
d['Unnamed: 8'] = None
d.to_csv(os.path.join(HERE, 'detect_dataset.csv'), index=False)
print('wrote', os.path.join(HERE, 'classData.csv'), 'and detect_dataset.csv', d.shape)
