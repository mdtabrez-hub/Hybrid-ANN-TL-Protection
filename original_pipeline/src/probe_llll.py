import numpy as np, pandas as pd, sys
sys.path.insert(0,'src')
from feature_engineering import build_rowwise_features, FEATURE_NAMES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os as _os
U=_os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")) + "/"
cd=pd.read_csv(U+"classData.csv")
vec=cd[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)
m=vec.isin(['0111','1111']); sub=cd[m].reset_index(drop=True)
y=(vec[m]=='1111').astype(int).to_numpy()
F=build_rowwise_features(sub)
print("n LLL =",(y==0).sum()," n LLLG =",(y==1).sum())

print("\n=== Is any single raw feature trivially separating? (AUC per feature) ===")
from sklearn.metrics import roc_auc_score
aucs={c: roc_auc_score(y,F[c]) for c in FEATURE_NAMES}
for c,a in sorted(aucs.items(), key=lambda kv:-abs(kv[1]-.5))[:10]:
    print(f"  {c:18s} AUC={a:.4f}")

print("\n=== RF feature importance ===")
Xtr,Xte,ytr,yte=train_test_split(F.to_numpy(),y,test_size=.2,random_state=42,stratify=y)
sc=MinMaxScaler().fit(Xtr)
rf=RandomForestClassifier(n_estimators=400,random_state=42,n_jobs=-1).fit(sc.transform(Xtr),ytr)
imp=sorted(zip(FEATURE_NAMES,rf.feature_importances_),key=lambda kv:-kv[1])[:8]
for c,i in imp: print(f"  {c:18s} {i:.4f}")
print(f"  RF acc = {accuracy_score(yte,rf.predict(sc.transform(Xte)))*100:.2f}%")

print("\n=== ROW-ORDER CHECK: are the two classes contiguous blocks in the file? ===")
idx=np.where(m)[0]
print("  first 10 original row indices of LLL :",idx[(y==0)][:10])
print("  first 10 original row indices of LLLG:",idx[(y==1)][:10])
print("  LLL  index range:",idx[y==0].min(),"-",idx[y==0].max())
print("  LLLG index range:",idx[y==1].min(),"-",idx[y==1].max())

print("\n=== Does RF survive when only PHYSICS features are used (no row-order proxy)? ===")
print("  |I0| alone:")
from sklearn.tree import DecisionTreeClassifier
for feats in [['I_zero_seq'],['I_zero_seq','abs_Ia','abs_Ib','abs_Ic'],
              ['Ia','Ib','Ic','Va','Vb','Vc']]:
    Xa=F[feats].to_numpy()
    xtr,xte,ya,yb=train_test_split(Xa,y,test_size=.2,random_state=42,stratify=y)
    r=RandomForestClassifier(n_estimators=400,random_state=42,n_jobs=-1).fit(xtr,ya)
    print(f"    {str(feats):55s} -> {accuracy_score(yb,r.predict(xte))*100:6.2f}%")

print("\n=== Sample rows: LLL vs LLLG side by side ===")
print(sub.assign(y=y).groupby('y')[['Ia','Ib','Ic','Va','Vb','Vc']].describe().T.round(3).to_string())
