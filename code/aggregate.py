"""Aggregate Phase 3 runs -> tables (CSV/MD) and figures. Independent reimplementation results only."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import json, glob, numpy as np, pandas as pd, itertools
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':7,'axes.titlesize':7.5,'legend.fontsize':6,'xtick.labelsize':6.5,'ytick.labelsize':6.5})
NAMES=['NoFault','LG','LL','LLG','LLL','LLLG']
rows=[]
for f in glob.glob('results/runs/*.json'):
    r=json.load(open(f)); c=r['cfg']
    key=dict(task=c['task'],data=c['data'],clip=c['clip'],scaler=c['scaler'],part=c['part'],gap=c.get('gap',0),shuffle=c.get('shuffle',1),drop_v0=c.get('drop_v0',0),seed=c['seed'])
    row=dict(**key,acc=r['acc'],merged=r.get('merged_acc',np.nan),n_test=r['n_test'],n_train=r['n_train'],epochs=r['epochs'])
    if c['task']=='class':
        for i,n in enumerate(NAMES):
            row[f'rec_{n}']=r['recall'][i]; row[f'prec_{n}']=r['precision'][i]; row[f'f1_{n}']=r['f1'][i]; row[f'sup_{n}']=r['support'][i]
        cm=np.array(r['cm']); row['pair_err']=cm[4,5]+cm[5,4]; row['nonpair_err']=cm.sum()-np.trace(cm)-row['pair_err']
        for k,v in r.get('acc_by_regime',{}).items(): row[f'accreg_{k}']=v
        for k,v in r.get('test_regime_counts',{}).items(): row[f'nreg_{k}']=v
    else:
        row['rec_fault']=r['recall'][1]; row['prec_fault']=r['precision'][1]
    row['cm']=json.dumps(r['cm']); row['excluded']=json.dumps(r.get('excluded_test_eval',[]))
    rows.append(row)
D=pd.DataFrame(rows); D.to_csv('results/all_runs.csv',index=False)
cond=['task','data','clip','scaler','part','gap','shuffle','drop_v0']
def ci(x):
    x=np.asarray(x,float); n=len(x); m=x.mean(); s=x.std(ddof=1) if n>1 else np.nan
    h=stats.t.ppf(.975,n-1)*s/np.sqrt(n) if n>1 else np.nan; return m,s,m-h,m+h,n
S=[]
for k,g in D.groupby(cond,dropna=False):
    m,s,lo,hi,n=ci(g.acc); rec={c:v for c,v in zip(cond,k)}
    rec.update(n_seeds=n,acc_mean=m,acc_sd=s,acc_ci_lo=lo,acc_ci_hi=hi,acc_min=g.acc.min(),acc_max=g.acc.max(),seeds=';'.join(f'{a:.4f}' for a in g.sort_values('seed').acc),
               n_test=int(g.n_test.iloc[0]),n_train=int(g.n_train.iloc[0]),epochs_mean=g.epochs.mean())
    if k[0]=='class':
        rec.update(merged_mean=g.merged.mean(),merged_sd=g.merged.std(ddof=1),pair_err_mean=g.pair_err.mean(),nonpair_err_mean=g.nonpair_err.mean(),pair_acc=1-g.pair_err.mean()/(g['sup_LLL'].iloc[0]+g['sup_LLLG'].iloc[0]))
        for n_ in NAMES:
            rec[f'rec_{n_}_mean']=g[f'rec_{n_}'].mean(); rec[f'rec_{n_}_sd']=g[f'rec_{n_}'].std(ddof=1); rec[f'rec_{n_}_min']=g[f'rec_{n_}'].min(); rec[f'rec_{n_}_max']=g[f'rec_{n_}'].max()
            rec[f'prec_{n_}_mean']=g[f'prec_{n_}'].mean(); rec[f'f1_{n_}_mean']=g[f'f1_{n_}'].mean(); rec[f'sup_{n_}']=int(g[f'sup_{n_}'].iloc[0])
        cms=np.sum([np.array(json.loads(x)) for x in g.cm],axis=0); rec['cm_sum']=json.dumps(cms.tolist())
        for col in [c for c in g.columns if c.startswith('accreg_')]: rec[col+'_mean']=g[col].mean()
        for col in [c for c in g.columns if c.startswith('nreg_')]: rec[col]=g[col].iloc[0]
    else:
        rec.update(rec_fault_mean=g.rec_fault.mean(),prec_fault_mean=g.prec_fault.mean())
    S.append(rec)
S=pd.DataFrame(S); S.to_csv('results/summary_by_condition.csv',index=False)
# paired comparisons (same seeds)
def sel(**kw):
    base=dict(task='class',data='E0',clip='train',scaler='minmax',gap=0,shuffle=1,drop_v0=0); base.update(kw)
    g=D.copy()
    for k,v in base.items(): g=g[g[k]==v]
    return g.set_index('seed').sort_index()
P=[]
def pair(name,a,b,metric='acc'):
    j=a[[metric]].join(b[[metric]],lsuffix='_a',rsuffix='_b',how='inner'); dlt=j[f'{metric}_a']-j[f'{metric}_b']
    m,s,lo,hi,n=ci(dlt); P.append(dict(comparison=name,metric=metric,n=n,mean_diff=m,sd=s,ci_lo=lo,ci_hi=hi,n_negative=int((dlt<0).sum()),per_seed=';'.join(f'{x:+.4f}' for x in dlt)))
for d_ in ['E0','E1','E2']:
    for p in ['tail','head','mid']: pair(f'{d_}: {p} - random',sel(data=d_,part=p),sel(data=d_,part='random'))
    pair(f'{d_}: tail - mid',sel(data=d_,part='tail'),sel(data=d_,part='mid'))
for p in ['random','head','mid','tail']:
    pair(f'{p}: E1 - E0',sel(data='E1',part=p),sel(data='E0',part=p)); pair(f'{p}: E2 - E0',sel(data='E2',part=p),sel(data='E0',part=p))
    pair(f'{p}: global clip - train clip',sel(part=p,clip='global'),sel(part=p))
for p in ['random','mid','tail']:
    pair(f'{p}: robust - minmax',sel(part=p,scaler='robust'),sel(part=p)); pair(f'{p}: robust(no clip) - minmax',sel(part=p,scaler='robust',clip='none'),sel(part=p))
for p in ['mid','tail']: pair(f'{p}: guard100 - no guard',sel(part=p,gap=100),sel(part=p))
pair('tail: unshuffled val - shuffled',sel(part='tail',shuffle=0),sel(part='tail'))
pair('random: 23 feats (no V0) - 24',sel(part='random',drop_v0=1),sel(part='random'))
for p in ['random','tail']: pair(f'tail-vs-random merged E0' if p=='tail' else 'merged dummy',sel(part=p),sel(part='random'),'merged')
P=pd.DataFrame(P); P=P[P.comparison!='merged dummy']; P.to_csv('results/paired_comparisons.csv',index=False)
# excluded-row (clearing) evaluation from E1 runs
ex=[]
for _,r in D[(D.data!='E0')&(D.task=='class')].iterrows():
    for e in json.loads(r.excluded): ex.append(dict(data=r.data,part=r.part,seed=r.seed,clip=r.clip,**e))
EX=pd.DataFrame(ex); EX.to_csv('results/excluded_rows_predictions.csv.gz',index=False)
print('runs',len(D),'conditions',len(S),'paired',len(P),'excluded-eval rows',len(EX))

# ---------------- figures ----------------
def cm_of(**kw):
    g=S.copy(); base=dict(task='class',data='E0',clip='train',scaler='minmax',gap=0,shuffle=1,drop_v0=0); base.update(kw)
    for k,v in base.items(): g=g[g[k]==v]
    return np.array(json.loads(g.iloc[0].cm_sum))
fig,ax=plt.subplots(1,2,figsize=(3.45,1.85))
lab=['NF','LG','LL','LLG','LLL','LLLG']
for A,(p,t) in zip(ax,[('random','(a) random split'),('tail','(b) tail hold-out')]):
    cm=cm_of(part=p); rn=cm/cm.sum(1,keepdims=True)
    A.imshow(rn,cmap='Greys',vmin=0,vmax=1)
    for i in range(6):
        for j in range(6):
            if rn[i,j]>=0.005: A.text(j,i,f'{100*rn[i,j]:.0f}',ha='center',va='center',fontsize=5,color='w' if rn[i,j]>.55 else 'k')
    A.set_xticks(range(6)); A.set_yticks(range(6)); A.set_xticklabels(lab,fontsize=5,rotation=90); A.set_yticklabels(lab,fontsize=5)
    A.set_title(t,fontsize=6.5); A.set_xlabel('Predicted',fontsize=6)
ax[0].set_ylabel('True',fontsize=6)
plt.tight_layout(w_pad=0.4); plt.savefig('figures/fig3_confusion.png',dpi=300); plt.close()
fig,ax=plt.subplots(1,1,figsize=(3.45,1.9))
parts=['random','head','mid','tail']; data=['E0','E1','E2']; off={'E0':-0.2,'E1':0,'E2':0.2}; col={'E0':'k','E1':'#c00000','E2':'#5b9bd5'}
nm={'E0':'E0 original labels','E1':'E1 inconsistent rows removed','E2':'E2 all post-command rows removed'}
for dd in data:
    for i,p in enumerate(parts):
        g=D[(D.task=='class')&(D.data==dd)&(D['clip']=='train')&(D.scaler=='minmax')&(D.part==p)&(D.gap==0)&(D.shuffle==1)&(D.drop_v0==0)]
        ax.scatter(np.full(len(g),i+off[dd]),100*g.acc,s=6,color=col[dd],alpha=.6,lw=0)
        ax.errorbar(i+off[dd],100*g.acc.mean(),yerr=100*g.acc.std(ddof=1),fmt='_',color=col[dd],ms=8,capsize=2,lw=.8,label=nm[dd] if i==0 else None)
ax.set_xticks(range(4)); ax.set_xticklabels(['random','head 20 %','middle 20 %','tail 20 %'],fontsize=6.5); ax.set_ylabel('Six-class accuracy (%)')
ax.legend(fontsize=5.5,frameon=False,loc='lower left'); ax.grid(axis='y',alpha=.3)
plt.tight_layout(); plt.savefig('figures/fig4_partitions.png',dpi=300); plt.close()
print('figures written')
