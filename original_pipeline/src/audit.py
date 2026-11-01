import pandas as pd, numpy as np, json, hashlib
import os as _os
U=_os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")) + "/"
def sha(p):
    h=hashlib.sha256()
    h.update(open(p,'rb').read()); return h.hexdigest()[:16]
cd=pd.read_csv(U+"classData.csv"); dd=pd.read_csv(U+"detect_dataset.csv")
print("=== classData.csv ===")
print("sha256[:16]",sha(U+"classData.csv"))
print("shape",cd.shape); print("cols",list(cd.columns))
print("dtypes\n",cd.dtypes)
print("nulls\n",cd.isna().sum().to_dict())
print("full-dup rows:",cd.duplicated().sum())
print("\n=== detect_dataset.csv ===")
print("sha256[:16]",sha(U+"detect_dataset.csv"))
print("shape",dd.shape); print("cols",list(dd.columns))
print("nulls\n",dd.isna().sum().to_dict())
print("all-empty cols:",[c for c in dd.columns if dd[c].isna().all()])
print("full-dup rows:",dd.duplicated().sum())
print("\n=== LABEL VECTORS (from CSV, order G,C,B,A) ===")
lab=cd[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)
vc=lab.value_counts()
print(vc.to_string()); print("total",vc.sum())
print("\nunique label values per col:",{c:sorted(cd[c].unique().tolist()) for c in ['G','C','B','A']})
print("\n=== DETECTION LABELS ===")
oc=[c for c in dd.columns if 'utput' in c or c.strip() in ('S','Output (S)')]
print("output col candidates",oc)
print(dd[oc[0]].value_counts().to_string())
print("\n=== MEASUREMENT RANGES (classData) ===")
meas=['Ia','Ib','Ic','Va','Vb','Vc']
print(cd[meas].describe().T.to_string())
print("\n=== IQR + percentile (classData) ===")
for c in meas:
    q1,q3=cd[c].quantile([.25,.75]); iqr=q3-q1
    lo,hi=q1-1.5*iqr,q3+1.5*iqr
    n=((cd[c]<lo)|(cd[c]>hi)).sum()
    p1,p99=cd[c].quantile([.01,.99])
    print(f"{c}: IQRflags={n}  p1={p1:.6f}  p99={p99:.6f}")
