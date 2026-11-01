"""
Does the prescribed per-channel percentile clipping damage the earth discriminant?

I_0 = (Ia+Ib+Ic)/3 is a DERIVED sum. Clipping Ia, Ib, Ic independently breaks the
balance between them and manufactures a residual current that is not present in the
measurement. This ablation varies only whether clipping is applied, and where.
Topology, initialisation, optimiser and schedule are the exact Blueprint model.
"""
import os,sys; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import numpy as np, pandas as pd
sys.path.insert(0,'src')
from feature_engineering import build_rowwise_features, FEATURE_NAMES
from train_ann import VEC2CLASS, CLASS_ORDER, keras_mlp, clip_1_99, MEAS, SEED
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf

import os as _os
U=_os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")) + "/"
cd=pd.read_csv(U+"classData.csv")
vec=cd[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)
y=pd.Categorical(vec.map(VEC2CLASS),categories=CLASS_ORDER).codes

def clip_features(F):
    """Clip the 24 engineered features instead of the raw phase channels."""
    out=F.copy()
    for c in out.columns:
        lo,hi=out[c].quantile([.01,.99]); out[c]=out[c].clip(lo,hi)
    return out

VARIANTS={
 "A. clip raw channels, MinMax (Blueprint, as specified)":
    (build_rowwise_features(clip_1_99(cd,MEAS)), MinMaxScaler),
 "B. no clipping, MinMax":
    (build_rowwise_features(cd), MinMaxScaler),
 "C. clip engineered features, MinMax":
    (clip_features(build_rowwise_features(cd)), MinMaxScaler),
 "D. no clipping, Robust":
    (build_rowwise_features(cd), RobustScaler),
 "E. clip engineered features, Robust":
    (clip_features(build_rowwise_features(cd)), RobustScaler),
}

rows=[]
for tag,(F,Sc) in VARIANTS.items():
    X=F.to_numpy(float)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    s=Sc().fit(Xtr)
    tf.keras.utils.set_random_seed(SEED)
    m=keras_mlp(X.shape[1],6,"softmax","sparse_categorical_crossentropy")
    m.fit(s.transform(Xtr),ytr,epochs=100,batch_size=32,validation_split=.15,verbose=0,
          shuffle=True,callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=.5,patience=5,min_lr=1e-6)])
    p=m.predict(s.transform(Xte),verbose=0).argmax(1)
    acc=accuracy_score(yte,p); _,rc,_,_=precision_recall_fscore_support(yte,p,zero_division=0)
    print(f"{tag:52s} acc={acc*100:6.2f}%  LLL rec={rc[4]*100:6.2f}%  LLLG rec={rc[5]*100:6.2f}%")
    rows.append(dict(variant=tag,accuracy=round(acc*100,4),
                     LLL_recall=round(rc[4]*100,4),LLLG_recall=round(rc[5]*100,4)))
pd.DataFrame(rows).to_csv('out/clipping_ablation.csv',index=False)
print("\n-> out/clipping_ablation.csv")
