"""Every dataset-level number cited in the revised manuscript, computed from the publisher's files."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np, json
R=DATA; Ts=50e-6
c=pd.read_csv(R+'classData.csv'); c['vec']=c[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)
d=pd.read_excel(R+'detect_dataset.xlsx'); reg=pd.read_csv('results/detect_row_regimes.csv')['regime'].values
from scipy.spatial import cKDTree
RAW=['Ia','Ib','Ic','Va','Vb','Vc']; sc=np.array([1e-3]*3+[1]*3); dist,tix=cKDTree(d[RAW].values*sc).query(c[RAW].values*sc)
c['t']=tix; c['regime']=reg[tix]; c.loc[c.vec=='0000','regime']='healthy'
F={}
F['n_class_rows']=len(c); F['n_detect_rows']=len(d); F['n_detect_fault']=int(d['Output (S)'].sum()); F['n_detect_healthy']=int((d['Output (S)']==0).sum())
F['class_subset_of_detect_maxdist']=float(dist.max()); F['Ts_us']=50; F['stop_s']=0.6; F['samples_per_cycle']=400
F['counts']=c.vec.value_counts().to_dict()
blk={}
for v,g in c.groupby('vec'):
    t=g.t.values; blk[v]=dict(start_ms=round(t.min()*Ts*1e3,2),end_ms=round(t.max()*Ts*1e3,2),cycles=round(len(t)*Ts*50,2),contiguous=bool(np.all(np.diff(t)==1)))
F['blocks']=blk
F['healthy_block_ms']=[blk['0000']['start_ms'],blk['0000']['end_ms']]
F['regime_counts']=c[c.vec!='0000'].regime.value_counts().to_dict()
F['regime_counts_by_class']={v:g.regime.value_counts().to_dict() for v,g in c[c.vec!='0000'].groupby('vec')}
I0=(c.Ia+c.Ib+c.Ic)/3; c['I0']=I0
F['V_sum_max_abs']=float(c[['Va','Vb','Vc']].sum(axis=1).abs().max())
acf={}
for v,g in c.groupby('vec'):
    x=g.Ia.values-g.Ia.mean(); acf[v]=round(float(np.dot(x[:-1],x[1:])/np.dot(x,x)),5)
F['acf_lag1_Ia']=acf; F['acf_min']=min(acf.values()); F['acf_max']=max(acf.values())
st={}
for v,g in c.groupby('vec'):
    a=g.I0.abs(); st[v]=dict(mean_abs_I0=round(a.mean(),3),cv=round(a.std(ddof=0)/a.mean(),3),rms=[round(float(np.sqrt((g[k]**2).mean())),1) for k in ['Ia','Ib','Ic']])
F['class_stats']=st
L=c[c.vec=='0111']; G=c[c.vec=='1111']
F['LLL_I0_signed_range']=[round(L.I0.min(),4),round(L.I0.max(),4)]; F['LLL_band_width']=round(L.I0.abs().max()-L.I0.abs().min(),4)
F['LLLG_in_LLL_band']=int(((G.I0.abs()>=L.I0.abs().min())&(G.I0.abs()<=L.I0.abs().max())).sum())
big=G.I0.abs()>L.I0.abs().max(); F['LLLG_above_LLL_max']=int(big.sum()); F['LLLG_above_LLL_max_all_partial_or_post']=bool(G[big].regime.isin(['partial_interruption','post_interruption']).all())
F['LLLG_sustained_mean_abs_I0']=round(G[G.regime=='sustained'].I0.abs().mean(),4)
F['LLLG_partial_mean_abs_I0']=round(G[G.regime=='partial_interruption'].I0.abs().mean(),1)
F['I0_span']=round(I0.max()-I0.min(),2)
lo=c[RAW].quantile(.01); hi=c[RAW].quantile(.99); cc=c[RAW].clip(lo,hi,axis=1); I0c=(cc.Ia+cc.Ib+cc.Ic)/3
Lc=I0c[c.vec=='0111'].abs(); alt=(cc.loc[L.index,['Ia','Ib','Ic']]!=c.loc[L.index,['Ia','Ib','Ic']]).any(axis=1)
F['clip_global_limits']={k:[round(lo[k],3),round(hi[k],3)] for k in RAW}
F['clip_LLL_rows_altered']=int(alt.sum()); F['clip_LLL_band']=[round(Lc.min(),4),round(Lc.max(),4)]
Gc=I0c[c.vec=='1111'].abs(); F['clip_LLLG_inside']=int(((Gc>=Lc.min())&(Gc<=Lc.max())).sum())
V0c=cc[['Va','Vb','Vc']].sum(axis=1)/3; F['V0_after_clip_nonzero_rows']=int((V0c.abs()>1e-6).sum()); F['V0_after_clip_max']=round(float(V0c.abs().max()),4)
pk=d[['Ia','Ib','Ic']].abs().max(axis=1); F['detect_median_peak_fault']=round(pk[d['Output (S)']==1].median(),1); F['detect_median_peak_healthy']=round(pk[d['Output (S)']==0].median(),1)
F['timeline']=json.load(open('results/event_timeline.json'))['200']
json.dump(F,open('results/data_facts.json','w'),indent=1,default=lambda o: o.item() if hasattr(o,'item') else str(o))
print(json.dumps({k:F[k] for k in ['regime_counts','acf_min','acf_max','LLL_I0_signed_range','LLLG_above_LLL_max','LLLG_above_LLL_max_all_partial_or_post','LLLG_sustained_mean_abs_I0','clip_LLL_rows_altered','clip_LLLG_inside','V0_after_clip_nonzero_rows']},default=str))
# ---- label-boundary and threshold-sensitivity facts (from event_timeline.json; all thresholds)
TL=json.load(open('results/event_timeline.json'))
t2=TL['200']; ev=[k for k in t2]
F['label_start_delay_ms']=[round(min(t2[k]['label_start_ms']-t2[k]['switch_on_ms'] for k in ev if k!='0110'),2),round(max(t2[k]['label_start_ms']-t2[k]['switch_on_ms'] for k in ev if k!='0110'),2)]
F['label_start_delay_LL_ms']=round(t2['0110']['label_start_ms']-t2['0110']['switch_on_ms'],2)
last=lambda e: max(v for kk,v in e.items() if kk.startswith('int_') and v==v)
F['label_end_after_lastpole_ms']=[round(min(t2[k]['label_end_ms']-last(t2[k]) for k in ev),2),round(max(t2[k]['label_end_ms']-last(t2[k]) for k in ev),2)]
F['partial_count_maxdiff_150_300']=int(max(abs(TL['150'][k]['n_partial']-TL['300'][k]['n_partial']) for k in ev))
posts=[TL[t][k]['n_post'] for t in ('150','300') for k in ev]; F['post_count_range_150_300']=[int(min(posts)),int(max(posts))]
S_=d['Output (S)'].values; ch=np.flatnonzero(np.diff(S_))+1
bc=[float(np.abs(d[['Ia','Ib','Ic']].values[i]).max()) for j in ch for i in (j-1,j)]
F['boundary_current_range_A']=[round(min(bc)),round(max(bc))]
h=d.loc[reg=='healthy',['Ia','Ib','Ic']].abs().max(axis=1)
F['healthy_peak_p99_A']=round(float(h.quantile(.99)),1); F['healthy_peak_median_A']=round(float(h.median()),1)
fp=d.loc[S_==1,['Ia','Ib','Ic']].abs().max(axis=1); F['fault_peak_max_A']=round(float(fp.max()),0)
json.dump(F,open('results/data_facts.json','w'),indent=1,default=lambda o: o.item() if hasattr(o,'item') else str(o))
print({k:F[k] for k in ['label_start_delay_ms','label_start_delay_LL_ms','label_end_after_lastpole_ms','partial_count_maxdiff_150_300','post_count_range_150_300','boundary_current_range_A','healthy_peak_p99_A','healthy_peak_median_A','fault_peak_max_A']})
