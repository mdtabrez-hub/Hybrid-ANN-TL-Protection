"""Physical event timeline of the published simulation (independent audit; public data only).
For each fault event: switch-on/off command times (from ANN_project.slx), the instant each faulted
phase's fault current is interrupted, and where the publisher's fault label ends."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np, json
R=DATA
d=pd.read_excel(R+'detect_dataset.xlsx'); Ts=50e-6; t=np.arange(len(d))*Ts
I=d[['Ia','Ib','Ic']].values; V=d[['Va','Vb','Vc']].values; S=d['Output (S)'].values
ev={'LG':(0.025,0.075,'A',True),'LLG':(0.125,0.175,'AB',True),'LL':(0.225,0.275,'BC',False),'LLL':(0.325,0.375,'ABC',False),'LLLG':(0.425,0.475,'ABC',True)}
# healthy reference: pre-fault interval 0..25 ms is too short for a full cycle-average? 500 samples = 1.25 cycles -> use peak envelope
pre_peak=np.abs(I[:500]).max(0); print('pre-fault peak |I| per phase (A):',pre_peak.round(1))
out={}
lab_runs=np.r_[0,np.flatnonzero(np.diff(S))+1,len(S)]
for k,(ton,toff,ph,g) in ev.items():
    i_on,i_off=int(round(ton/Ts)),int(round(toff/Ts))
    # label run containing ton
    a=[x for x in lab_runs if x<=i_on+20][-1]; b=[x for x in lab_runs if x>i_on+20][0]-1
    res={'switch_on_ms':ton*1e3,'switch_off_ms':toff*1e3,'label_start_ms':a*Ts*1e3,'label_end_ms':b*Ts*1e3}
    # per-phase interruption: last sample after toff where |I| exceeds 2x pre-fault peak of that phase
    for p in ph:
        j='ABC'.index(p); thr=2.0*max(pre_peak[j],60)
        seg=np.abs(I[i_off:i_off+800,j]); above=np.flatnonzero(seg>thr)
        res[f'interrupt_{p}_ms']=(i_off+(above[-1] if len(above) else 0))*Ts*1e3
    if g:
        i0=np.abs(I.sum(1))/3; seg=i0[i_off:i_off+800]; above=np.flatnonzero(seg>5)
        res['I0_last_above5A_ms']=(i_off+(above[-1] if len(above) else 0))*Ts*1e3
    last=max(v for kk,v in res.items() if kk.startswith('interrupt_'))
    res['last_pole_interrupt_ms']=last
    res['rows_on_to_off']=i_off-a; res['rows_off_to_lastpole']=int(round(last/1e3/Ts))-i_off; res['rows_after_lastpole_in_label']=b-int(round(last/1e3/Ts))
    out[k]=res
print(json.dumps(out,indent=1,default=float))
json.dump(out,open(ROOT+"/results/event_timeline.json","w"),indent=1,default=float)
