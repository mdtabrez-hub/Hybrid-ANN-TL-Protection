"""Physically defined regimes for every labelled-fault row (independent audit, public data).
Interruption of a faulted phase = first sample at/after the switch-off command from which |I_phase| stays
below 200 A for the following 10 ms (200 samples). 200 A is ~3x the largest healthy load peak (~90 A) and
far below fault-current peaks (~880 A). Sensitivity to the 200 A choice is reported."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np, json
R=DATA; Ts=50e-6
d=pd.read_excel(R+'detect_dataset.xlsx'); I=d[['Ia','Ib','Ic']].values; S=d['Output (S)'].values
ev={'1001':(0.025,0.075,'A'),'1011':(0.125,0.175,'AB'),'0110':(0.225,0.275,'BC'),'0111':(0.325,0.375,'ABC'),'1111':(0.425,0.475,'ABC')}
runs=np.r_[0,np.flatnonzero(np.diff(S))+1,len(S)]
def interrupt(j,i_off,thr):
    a=np.abs(I[:,j])
    for k in range(i_off,i_off+600):
        if a[k:k+200].max()<thr: return k
    return None
regime=np.array(['healthy']*len(S),dtype=object); info={}
for thr in (150,200,300):
    tab={}
    for v,(ton,toff,ph) in ev.items():
        i_on,i_off=int(round(ton/Ts)),int(round(toff/Ts))
        st=[r for r in runs if r<=i_off and S[r]==1 and (r==0 or S[r-1]==0)][-1]; en=[r for r in runs if r>st][0]-1
        ints={p:interrupt('ABC'.index(p),i_off,thr) for p in ph}
        first,last=min(ints.values()),max(ints.values())
        tab[v]=dict(label_start_ms=st*Ts*1e3,switch_on_ms=ton*1e3,switch_off_ms=toff*1e3,label_end_ms=en*Ts*1e3,
            **{f'int_{p}_ms':i*Ts*1e3 for p,i in ints.items()},
            n_label=en-st+1, n_sustained=i_off-st, n_clear_alltypes=first-i_off, n_partial=last-first, n_post=en-last+1)
        if thr==200:
            regime[st:i_off]='sustained'; regime[i_off:first]='clearing_sametype'; regime[first:last]='partial_interruption'; regime[last:en+1]='post_interruption'
    info[thr]=tab
print(pd.DataFrame(info[200]).T.round(2).to_string())
for thr in (150,300): print(f'\nthr={thr}: partial/post rows', {v:(t['n_partial'],t['n_post']) for v,t in info[thr].items()})
# label rule inference: max |I| at label transitions
print('\nLabel transitions (max|I| across phases just before -> just after):')
for r in runs[1:-1]:
    print(f"  t={r*Ts*1e3:7.2f} ms S {S[r-1]}->{S[r]}: {np.abs(I[r-1]).max():6.1f} -> {np.abs(I[r]).max():6.1f} A")
d['regime']=regime; d[['regime']].to_csv(ROOT+'/results/detect_row_regimes.csv',index_label='row')
json.dump({str(k):v for k,v in info.items()},open(ROOT+'/results/event_timeline.json','w'),indent=1,default=float)
print('\nregime counts (fault rows):',pd.Series(regime[S==1]).value_counts().to_dict())
