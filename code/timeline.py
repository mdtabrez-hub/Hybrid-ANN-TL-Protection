import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np
R=DATA
c=pd.read_csv(R+'classData.csv'); d=pd.read_excel(R+'detect_dataset.xlsx')
cols=['Ia','Ib','Ic','Va','Vb','Vc']
c['vec']=c[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)
key=lambda df: [tuple(np.round(r,6)) for r in df[cols].values]
dk={k:i for i,k in enumerate(key(d))}
idx=np.array([dk.get(k,-1) for k in key(c)])
print('classData rows found in detect:', (idx>=0).sum(), 'of', len(c))
c['t_idx']=idx; Ts=50e-6
for v,g in c.groupby('vec',sort=False):
    t=g['t_idx'].values; ok=t[t>=0]
    contiguous=np.all(np.diff(ok)==1) if len(ok) else None
    print(f"{v}: n={len(g)} found={len(ok)} detect-index {ok.min()}..{ok.max()}  t={ok.min()*Ts:.5f}..{ok.max()*Ts:.5f}s  contiguous_in_time={contiguous}  monotone={np.all(np.diff(ok)>0)}")
# detect label runs
s=d['Output (S)'].values; ch=np.flatnonzero(np.diff(s))+1
print('detect S runs (start idx,t,value):',[(0,0.0,int(s[0]))]+[(int(i),round(i*Ts,5),int(s[i])) for i in ch])
# healthy block in classData: where in time?
h=c[c.vec=='0000']['t_idx'].values
print('healthy idx segments:',[(int(a),int(b)) for a,b in zip(np.r_[h[0],h[1:][np.diff(h)!=1]], np.r_[h[:-1][np.diff(h)!=1],h[-1]])][:20])
c.to_pickle(WORK+'/class_with_time.pkl'); d.to_pickle(WORK+'/detect.pkl')
