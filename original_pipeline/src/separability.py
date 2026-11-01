"""Why does classification cap near 85% while detection reaches 99.8%?"""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0,'src')
from feature_engineering import build_rowwise_features, MEAS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os as _os
U=_os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")) + "/"
cd=pd.read_csv(U+"classData.csv")
vec=cd[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)

print("=== 1. Zero-sequence current by class (the G-bit discriminator) ===")
I0=(cd[['Ia','Ib','Ic']].sum(axis=1)/3)
for v in ['0000','1001','0110','1011','0111','1111']:
    s=I0[vec==v].abs()
    print(f"  {v} ({ {'0000':'No fault','1001':'LG','0110':'LL','1011':'LLG','0111':'LLL','1111':'LLLG'}[v] :9s}) "
          f"mean={s.mean():8.2f}  median={s.median():8.2f}  p90={s.quantile(.9):8.2f}  max={s.max():8.2f}")

print("\n=== 2. Overlap of |I0| between LLL (0111) and LLLG (1111) ===")
a=I0[vec=='0111'].abs(); b=I0[vec=='1111'].abs()
print(f"  LLL  |I0|: mean {a.mean():.3f}  p99 {a.quantile(.99):.3f}")
print(f"  LLLG |I0|: mean {b.mean():.3f}  p99 {b.quantile(.99):.3f}")
thr=np.linspace(0,60,601); best=(0,0)
for t in thr:
    acc=((a<t).sum()+(b>=t).sum())/(len(a)+len(b))
    if acc>best[1]: best=(t,acc)
print(f"  best single-threshold |I0| separation: thr={best[0]:.2f} -> {best[1]*100:.2f}% "
      f"(chance = {max(len(a),len(b))/(len(a)+len(b))*100:.2f}%)")

print("\n=== 3. Dedicated LLL-vs-LLLG classifier on all 24 features ===")
m=vec.isin(['0111','1111'])
Xb=build_rowwise_features(cd[m]).to_numpy(); yb=(vec[m]=='1111').astype(int).to_numpy()
Xtr,Xte,ytr,yte=train_test_split(Xb,yb,test_size=.2,random_state=42,stratify=yb)
sc=MinMaxScaler().fit(Xtr)
rf=RandomForestClassifier(n_estimators=400,random_state=42,n_jobs=-1).fit(sc.transform(Xtr),ytr)
acc=accuracy_score(yte,rf.predict(sc.transform(Xte)))
print(f"  RandomForest (a far stronger model than the ANN): {acc*100:.2f}%")
print(f"  majority-class baseline:                          {max(np.bincount(yte))/len(yte)*100:.2f}%")

print("\n=== 4. Six-class accuracy if LLL and LLLG are MERGED ===")
cm=np.loadtxt('out/cm_D.csv',delimiter=',',dtype=int)
names=["No fault","LG (A-G)","LL (B-C)","LLG (A-B-G)","LLL","LLLG"]
print("  confusion matrix (rows=true, cols=pred):")
print("   "+"".join(f"{n[:9]:>11s}" for n in names))
for i,n in enumerate(names):
    print(f"  {n[:11]:<12s}"+"".join(f"{cm[i,j]:>11d}" for j in range(6)))
M=cm.copy()
Mm=np.zeros((5,5),dtype=int)
idx=[[0],[1],[2],[3],[4,5]]
for i,ri in enumerate(idx):
    for j,cj in enumerate(idx):
        Mm[i,j]=M[np.ix_(ri,cj)].sum()
print(f"\n  six-class accuracy        : {np.trace(M)/M.sum()*100:.2f}%")
print(f"  five-class (LLL+LLLG merged): {np.trace(Mm)/Mm.sum()*100:.2f}%")
print(f"  share of ALL errors that are LLL<->LLLG confusions: "
      f"{(M[4,5]+M[5,4])/(M.sum()-np.trace(M))*100:.1f}%")
