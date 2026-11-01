import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=DATA; d=pd.read_excel(R+'detect_dataset.xlsx'); Ts=50e-6; t=np.arange(len(d))*Ts*1e3
I=d[['Ia','Ib','Ic']].values; S=d['Output (S)'].values; I0=I.sum(1)/3
ev={'LG A-G':(25,75),'LLG AB-G':(125,175),'LL B-C':(225,275),'LLL':(325,375),'LLLG':(425,475)}
fig,ax=plt.subplots(5,2,figsize=(14,14))
for r,(k,(a,b)) in enumerate(ev.items()):
    for c,(lo,hi) in enumerate([(a-4,a+12),(b-6,b+14)]):
        m=(t>=lo)&(t<=hi); A=ax[r,c]
        for j,col in enumerate(['C0','C1','C2']): A.plot(t[m],I[m,j],col,lw=1,label='Iabc'[j*1+1] if False else ['Ia','Ib','Ic'][j])
        A.plot(t[m],3*I0[m],'k',lw=1.2,label='3I0'); A.axvline(a if c==0 else b,color='r',ls='--',label='switch cmd')
        lab=S[m]; A.fill_between(t[m],-950,950,where=lab==1,color='grey',alpha=.12,label='label=fault')
        A.set_title(f'{k}: {"inception" if c==0 else "clearing"}'); A.set_ylim(-950,950); A.grid(alpha=.3)
        if r==0 and c==0: A.legend(fontsize=7,ncol=3)
plt.tight_layout(); plt.savefig(ROOT+'/figures/audit_event_windows.png',dpi=90)
