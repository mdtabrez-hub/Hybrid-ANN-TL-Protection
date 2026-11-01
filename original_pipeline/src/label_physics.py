import pandas as pd, numpy as np
import os as _os
U=_os.environ.get("FAULT_DATA_DIR", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")) + "/"
cd=pd.read_csv(U+"classData.csv")
cd['vec']=cd[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)
print("Per-label-vector mean |current| and mean |voltage| by phase")
print("(fault current should be elevated in the FAULTED phases; voltage depressed)\n")
rows=[]
for v,g in cd.groupby('vec'):
    r={'vec':v,'n':len(g)}
    for c in ['Ia','Ib','Ic']: r['|'+c+'|']=g[c].abs().mean()
    for c in ['Va','Vb','Vc']: r['|'+c+'|']=g[c].abs().mean()
    r['|I0|']=(g[['Ia','Ib','Ic']].sum(axis=1)/3).abs().mean()
    rows.append(r)
df=pd.DataFrame(rows).set_index('vec')
print(df.round(4).to_string())
print("\nInterpretation per vector (order G,C,B,A):")
for v in df.index:
    ia,ib,ic=df.loc[v,'|Ia|'],df.loc[v,'|Ib|'],df.loc[v,'|Ic|']
    mx=max(ia,ib,ic); hot=[p for p,val in zip('ABC',[ia,ib,ic]) if val>0.5*mx]
    print(f"  {v}: n={int(df.loc[v,'n'])}  elevated-current phases -> {hot}   |I0|={df.loc[v,'|I0|']:.3f}")
