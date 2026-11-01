"""
Diagnostic ablation: why does the exact Blueprint pipeline mis-classify LLLG?
Hypothesis: MinMax scaling of I_zero_seq compresses the LLL/LLLG discriminant
into a sliver of [0,1], because LG/LLG set a dynamic range ~300x larger.
Only the SCALER is varied; the network topology is the exact Blueprint model.
"""
import os,sys,json; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np, pandas as pd
sys.path.insert(0,'src')
from feature_engineering import build_rowwise_features, FEATURE_NAMES, MEAS
from train_ann import VEC2CLASS, CLASS_ORDER, keras_mlp, clip_1_99, SEED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf

import os as _os
U=_os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")) + "/"
cd=pd.read_csv(U+"classData.csv")
vec=cd[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)
X=build_rowwise_features(clip_1_99(cd,MEAS)).to_numpy(float)
y=pd.Categorical(vec.map(VEC2CLASS),categories=CLASS_ORDER).codes
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)

j=FEATURE_NAMES.index('I_zero_seq')
mm=MinMaxScaler().fit(Xtr); Z=mm.transform(X)[:,j]
print("=== Scaled I_zero_seq occupancy of the [0,1] range, by class ===")
for i,n in enumerate(CLASS_ORDER):
    s=Z[y==i]; print(f"  {n:12s} min={s.min():.6f} med={np.median(s):.6f} max={s.max():.6f} span={s.max()-s.min():.6f}")
lll,lllg=Z[y==4],Z[y==5]
print(f"\n  LLL vs LLLG separation in scaled units: "
      f"|median diff| = {abs(np.median(lll)-np.median(lllg)):.6f} of the full [0,1] range")
print(f"  float32 has ~1e-7 resolution, so the signal survives numerically but is")
print(f"  ~{1/max(abs(np.median(lll)-np.median(lllg)),1e-9):.0f}x smaller than the features the network sees at O(1).\n")

def signed_log(A):
    return np.sign(A)*np.log1p(np.abs(A))

def run(tag, ftr):
    Xa,Xb=ftr(Xtr),ftr(Xte)
    tf.keras.utils.set_random_seed(SEED)
    m=keras_mlp(X.shape[1],6,"softmax","sparse_categorical_crossentropy")
    m.fit(Xa,ytr,epochs=100,batch_size=32,validation_split=.15,verbose=0,shuffle=True,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(factor=.5,patience=5,min_lr=1e-6)])
    p=m.predict(Xb,verbose=0).argmax(1)
    acc=accuracy_score(yte,p)
    _,rc,_,_=precision_recall_fscore_support(yte,p,zero_division=0)
    print(f"  {tag:46s} acc={acc*100:6.2f}%   LLL recall={rc[4]*100:6.2f}%  LLLG recall={rc[5]*100:6.2f}%")
    return dict(variant=tag,accuracy=round(acc*100,4),
                LLL_recall=round(rc[4]*100,4),LLLG_recall=round(rc[5]*100,4))

print("=== Six-class accuracy, exact Blueprint topology, scaler varied ===")
res=[]
s1=MinMaxScaler().fit(Xtr);              res.append(run("MinMax  (Blueprint, as specified)", s1.transform))
s2=StandardScaler().fit(Xtr);            res.append(run("StandardScaler (zero-mean unit-var)", s2.transform))
s3=RobustScaler().fit(Xtr);              res.append(run("RobustScaler (median/IQR)", s3.transform))
Ltr=signed_log(Xtr); s4=MinMaxScaler().fit(Ltr)
res.append(run("signed-log -> MinMax", lambda A: s4.transform(signed_log(A))))
pd.DataFrame(res).to_csv('out/scaling_ablation.csv',index=False)
print("\n-> out/scaling_ablation.csv")
