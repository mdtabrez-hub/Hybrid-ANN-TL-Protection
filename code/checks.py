import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np
from scipy.spatial import cKDTree
c=pd.read_pickle(WORK+'/class_with_time.pkl'); d=pd.read_pickle(WORK+'/detect.pkl')
cols=['Ia','Ib','Ic','Va','Vb','Vc']; Ts=50e-6; f=50
# tolerant match
sc=np.array([1/1000]*3+[1]*3)
tree=cKDTree(d[cols].values*sc); dist,ix=tree.query(c[cols].values*sc)
c['t_idx']=ix; print('max NN dist (scaled):',dist.max(), ' rows with dist>1e-4:',(dist>1e-4).sum())
order=['1001','1011','0110','0111','1111','0000']; name=dict(zip(order,['LG(A-G)','LLG(AB-G)','LL(B-C)','LLL','LLLG','Healthy']))
sw={'1001':(0.025,0.075),'1011':(0.125,0.175),'0110':(0.225,0.275),'0111':(0.325,0.375),'1111':(0.425,0.475)}
print('\n== Blocks on the timeline')
for v in order:
    t=c.loc[c.vec==v,'t_idx'].values
    print(f"{name[v]:10s} n={len(t)} idx {t.min()}-{t.max()} ({t.min()*Ts*1e3:.2f}-{t.max()*Ts*1e3:.2f} ms) span={(t.max()-t.min()+1)} contiguous={np.all(np.diff(t)==1)} cycles={(len(t))*Ts*f:.2f}", end='')
    if v in sw:
        a,b=sw[v]; post=(t*Ts>b).sum(); print(f"  switch {a*1e3:.0f}-{b*1e3:.0f} ms; rows after switch-off={post} ({post/len(t)*100:.1f}%); tail20%={int(round(0.2*len(t)))}")
    else: print()
# voltages: line-to-line?
s=c[['Va','Vb','Vc']].sum(axis=1); print('\nmax |Va+Vb+Vc| =',s.abs().max(), ' median',s.abs().median())
# samples per cycle: zero crossings of healthy Va in detect pre-fault 0..503
va=d['Va'].values[:504]; zc=np.flatnonzero(np.diff(np.sign(va))>0); print('pre-fault +zero crossings idx',zc,' spacing',np.diff(zc))
# lag-1 acf within each block
print('\n== lag-1 autocorrelation of Ia per block')
for v in order:
    x=c.loc[c.vec==v,'Ia'].values; x=x-x.mean(); print(name[v], round(float(np.dot(x[:-1],x[1:])/np.dot(x,x)),6))
print('cos(2pi/400)=',np.cos(2*np.pi/400))
# Table 3 quantities
print('\n== Table 3 check')
c['I0']=(c.Ia+c.Ib+c.Ic)/3
for v in order:
    g=c[c.vec==v]; a=g.I0.abs()
    print(f"{name[v]:10s} mean|Ia|,|Ib|,|Ic| = {g.Ia.abs().mean():.0f}/{g.Ib.abs().mean():.0f}/{g.Ic.abs().mean():.0f}  mean|I0|={a.mean():.3f}  CV(|I0|) pop={a.std(ddof=0)/a.mean():.3f} samp={a.std(ddof=1)/a.mean():.3f}  rms(Ia..)={np.sqrt((g[['Ia','Ib','Ic']]**2).mean()).round(1).tolist()}")
lll=c[c.vec=='0111'].I0.abs(); lllg=c[c.vec=='1111'].I0.abs()
lo,hi=lll.min(),lll.max(); print(f"\nLLL band [{lo:.4f},{hi:.4f}] width {hi-lo:.4f}; LLLG inside band: {((lllg>=lo)&(lllg<=hi)).sum()}; LLLG below LLL max: {(lllg<hi).mean()*100:.1f}%")
h=c[c.vec=='0000']; ir=np.sqrt(((h[['Ia','Ib','Ic']]**2).sum(axis=1)/3)); print('healthy |I0|/Irms ratio (mean|I0|/mean Irms):', h.I0.abs().mean()/ir.mean())
g=c[c.vec=='0111']; ir=np.sqrt(((g[['Ia','Ib','Ic']]**2).sum(axis=1)/3)); print('LLL ratio:', g.I0.abs().mean()/ir.mean())
print('zero-seq span |I0| all rows (unclipped):', c.I0.min(), c.I0.max(), c.I0.max()-c.I0.min(), ' |I0| span', c.I0.abs().max()-c.I0.abs().min())
c.to_pickle(WORK+'/class_full.pkl')
