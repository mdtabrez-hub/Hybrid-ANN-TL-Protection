import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np
c=pd.read_pickle(WORK+'/class_full.pkl'); d=pd.read_pickle(WORK+'/detect.pkl'); Ts=50e-6
order=['1001','1011','0110','0111','1111','0000']; name=dict(zip(order,['LG','LLG','LL','LLL','LLLG','Healthy']))
sw={'1001':0.075,'1011':0.175,'0110':0.275,'0111':0.375,'1111':0.475}
lllmax=c.loc[c.vec=='0111','I0'].abs().max()
g=c[c.vec=='1111'].reset_index(drop=True); big=g.I0.abs()>lllmax
pos=np.flatnonzero(big)/len(g)
print(f"LLLG rows with |I0|>LLL max: {big.sum()} ({big.mean()*100:.1f}%); positions in block (fraction): min {pos.min():.3f} max {pos.max():.3f}; share inside last 20%: {(pos>=0.8).mean()*100:.1f}%  share after switch-off: {(g.t_idx[big]*Ts>0.475).mean()*100:.1f}%")
print("LLLG |I0| mean head80 vs tail20:", g.I0.abs()[:int(0.8*len(g))].mean(), g.I0.abs()[int(0.8*len(g)):].mean())
print('LLL signed I0 mean/min/max:', c.loc[c.vec=='0111','I0'].agg(['mean','min','max']).round(4).tolist())
print('\n== head (first 80%) vs tail (last 20%) per block: RMS of phase currents, mean|I0|')
for v in order:
    x=c[c.vec==v].reset_index(drop=True); k=int(round(0.8*len(x)))
    def s(y): return np.sqrt((y[['Ia','Ib','Ic']]**2).mean()).round(0).tolist(), round(y.I0.abs().mean(),2), np.sqrt((y[['Va','Vb','Vc']]**2).mean()).round(3).tolist()
    tail=x[k:]; post=(tail.t_idx*Ts>sw.get(v,9)).mean()*100
    print(f"{name[v]:8s} head I_rms {s(x[:k])[0]} |I0| {s(x[:k])[1]} V_rms {s(x[:k])[2]} || tail I_rms {s(tail)[0]} |I0| {s(tail)[1]} V_rms {s(tail)[2]} | tail rows after switch-off {post:.0f}%")
# clipping (whole-file percentiles) + effect on LLL
cols=['Ia','Ib','Ic','Va','Vb','Vc']
lo=c[cols].quantile(0.01); hi=c[cols].quantile(0.99); print('\n1st/99th pct (whole file):'); print(pd.DataFrame({'p1':lo,'p99':hi}).round(3).to_string())
cc=c.copy(); cc[cols]=cc[cols].clip(lo,hi,axis=1); cc['I0c']=(cc.Ia+cc.Ib+cc.Ic)/3
L=cc[cc.vec=='0111']; altered=(L[['Ia','Ib','Ic']]!=c.loc[L.index,['Ia','Ib','Ic']]).any(axis=1)
print(f"LLL rows altered by clip: {altered.sum()}; |I0c| range [{L.I0c.abs().min():.4f},{L.I0c.abs().max():.4f}]; max unclipped-row |I0| {L.I0c.abs()[~altered].max():.4f}")
LG=cc[cc.vec=='1111']; inside=((LG.I0c.abs()>=L.I0c.abs().min())&(LG.I0c.abs()<=L.I0c.abs().max())).sum(); print('LLLG inside clipped LLL band:',inside, f"({inside/len(LG)*100:.1f}%)")
q1=c[cols].quantile(.25); q3=c[cols].quantile(.75); iq=q3-q1
print('IQR flags:', (((c[cols]<q1-1.5*iq)|(c[cols]>q3+1.5*iq)).sum()).to_dict())
pk=d[['Ia','Ib','Ic']].abs().max(axis=1); print('detect median peak |I| faulted/healthy:', pk[d['Output (S)']==1].median().round(1), pk[d['Output (S)']==0].median().round(1))
h=c[c.vec=='0000']; print('healthy overall RMS current (all phases pooled):', np.sqrt((h[['Ia','Ib','Ic']]**2).values.mean()).round(2), ' mean|I| pooled', h[['Ia','Ib','Ic']].abs().values.mean().round(1))
# detect file run structure
s=d['Output (S)'].values; ch=np.r_[0,np.flatnonzero(np.diff(s))+1,len(s)]
print('detect runs (label,len):',[(int(s[a]),int(b-a)) for a,b in zip(ch[:-1],ch[1:])])
