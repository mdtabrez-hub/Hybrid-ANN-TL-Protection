"""Phase 3 experiment harness — INDEPENDENT REIMPLEMENTATION (Keras/TensorFlow) of the configuration published
in the manuscript (Table 6). It is NOT the author's pipeline and does not reproduce the author's runs.
Data: publisher's classData.csv / detect_dataset.xlsx (commit 4e094cf). One JSON result per run (resumable)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import os, sys, json, time, itertools, hashlib
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
tf.config.threading.set_inter_op_parallelism_threads(1); tf.config.threading.set_intra_op_parallelism_threads(2)
OUT=ROOT+'/results/runs'; os.makedirs(OUT,exist_ok=True)
R=DATA; Ts=50e-6
CLASSES=['0000','1001','0110','1011','0111','1111']; NAMES=['NoFault','LG','LL','LLG','LLL','LLLG']
RAW=['Ia','Ib','Ic','Va','Vb','Vc']
# ---------- data + physical regimes ----------
c=pd.read_csv(R+'classData.csv'); c['vec']=c[['G','C','B','A']].astype(int).astype(str).agg(''.join,axis=1)
d=pd.read_excel(R+'detect_dataset.xlsx'); reg=pd.read_csv(ROOT+'/results/detect_row_regimes.csv')['regime'].values
from scipy.spatial import cKDTree
sc=np.array([1e-3]*3+[1]*3); _,tix=cKDTree(d[RAW].values*sc).query(c[RAW].values*sc); c['t_idx']=tix; c['regime']=reg[tix]
c.loc[c.vec=='0000','regime']='healthy'
y=c.vec.map({v:i for i,v in enumerate(CLASSES)}).values
pos=np.zeros(len(c)); 
for v in CLASSES:
    m=np.flatnonzero(c.vec.values==v); pos[m]=np.arange(len(m))/len(m)
# physical topology of partial-interruption rows (from event timeline): LLG->A-G, LLL->A-B(not a label), LLLG->AB-G then A-G
tl=json.load(open(ROOT+'/results/event_timeline.json'))['200']
def phys(row):
    v,t=row.vec,row.t_idx*Ts*1e3
    if row.regime!='partial_interruption': return v
    if v=='1011': return '1001'
    if v=='1111': return '1011' if t<tl['1111']['int_B_ms'] else '1001'
    if v=='0111': return 'AB(0011)'
    return 'other'
c['phys']=c.apply(phys,axis=1)
def feats(df):
    I=df[['Ia','Ib','Ic']].values; V=df[['Va','Vb','Vc']].values; a=np.exp(2j*np.pi/3)
    I2=(I[:,0]+a**2*I[:,1]+a*I[:,2])/3; V2=(V[:,0]+a**2*V[:,1]+a*V[:,2])/3
    F=np.hstack([I,V,np.abs(I),np.abs(V),np.sqrt((I**2).mean(1,keepdims=True)),np.sqrt((V**2).mean(1,keepdims=True)),
        np.abs(I).max(1,keepdims=True),np.abs(V).max(1,keepdims=True),I.mean(1,keepdims=True),V.mean(1,keepdims=True),
        np.c_[I2.real,I2.imag],np.c_[V2.real,V2.imag],I.std(1,keepdims=True),(V*I).sum(1,keepdims=True)])
    return F  # 24 columns, Table 5 order; col 17 = V0 (index 17)
def build(seed,n_in,n_out):
    tf.keras.utils.set_random_seed(seed)
    he=tf.keras.initializers.HeNormal(seed=seed)
    m=tf.keras.Sequential([tf.keras.Input((n_in,)),tf.keras.layers.Dense(64,'relu',kernel_initializer=he),tf.keras.layers.Dropout(.2,seed=seed),
        tf.keras.layers.Dense(32,'relu',kernel_initializer=he),tf.keras.layers.Dropout(.2,seed=seed+1),tf.keras.layers.Dense(16,'relu',kernel_initializer=he),
        tf.keras.layers.Dense(n_out,'softmax' if n_out>1 else 'sigmoid')])
    m.compile(tf.keras.optimizers.Adam(1e-3),'sparse_categorical_crossentropy' if n_out>1 else 'binary_crossentropy',metrics=['accuracy'])
    return m
def split(part,seed,n,yy,pp,gap=0):
    idx=np.arange(n)
    if part=='random':
        tr,te=train_test_split(idx,test_size=0.2,stratify=yy,random_state=seed); return tr,te,np.zeros(n,bool)
    lo,hi={'head':(0,.2),'mid':(.4,.6),'tail':(.8,1.0)}[part]
    te=(pp>=lo)&(pp<hi); 
    guard=np.zeros(n,bool)
    if gap:
        for v in np.unique(yy):
            m=np.flatnonzero(yy==v); t=np.flatnonzero(te[m]); 
            if len(t): a,b=t.min(),t.max(); guard[m[max(0,a-gap):a]]=True; guard[m[b+1:b+1+gap]]=True
    return idx[~te & ~guard], idx[te], guard
def run(cfg):
    name='__'.join(f'{k}={v}' for k,v in sorted(cfg.items())); fn=f'{OUT}/{hashlib.md5(name.encode()).hexdigest()[:12]}.json'
    if os.path.exists(fn): return
    t0=time.time(); seed=cfg['seed']
    if cfg['task']=='detect':
        D=d.copy(); yy=D['Output (S)'].values; runs=np.r_[0,np.flatnonzero(np.diff(yy))+1,len(yy)]
        pp=np.zeros(len(D)); blk=np.zeros(len(D),int)
        for k,(a,b) in enumerate(zip(runs[:-1],runs[1:])): pp[a:b]=np.arange(b-a)/(b-a); blk[a:b]=k
        raw=D[RAW]; keep=np.ones(len(D),bool)
        tr,te,_=split(cfg['part'],seed,len(D),blk,pp)
    else:
        keep=np.ones(len(c),bool)
        if cfg['data']=='E1': keep=~c.regime.isin(['partial_interruption','post_interruption']).values
        if cfg['data']=='E2': keep=~c.regime.isin(['clearing_sametype','partial_interruption','post_interruption']).values
        tr,te,_=split(cfg['part'],seed,len(c),y,pos,cfg.get('gap',0))
        excl_te=te[~keep[te]]; tr=tr[keep[tr]]; te=te[keep[te]]; raw=c[RAW]; yy=y
    # clipping thresholds
    if cfg['clip']=='global': lo,hi=raw.quantile(.01),raw.quantile(.99)
    elif cfg['clip']=='train': lo,hi=raw.iloc[tr].quantile(.01),raw.iloc[tr].quantile(.99)
    rawc=raw if cfg['clip']=='none' else raw.clip(lo,hi,axis=1)
    X=feats(rawc)
    if cfg.get('drop_v0'): X=np.delete(X,17,axis=1)
    S=(MinMaxScaler if cfg['scaler']=='minmax' else RobustScaler)().fit(X[tr])
    Xs=S.transform(X)
    rng=np.random.default_rng(seed); trs=tr.copy()
    if cfg.get('shuffle',1): rng.shuffle(trs)          # Keras validation_split takes the LAST 15 %: shuffle first
    n_out=1 if cfg['task']=='detect' else 6
    m=build(seed,X.shape[1],n_out)
    cb=[tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(factor=.5,patience=5)]
    h=m.fit(Xs[trs],yy[trs],batch_size=32,epochs=100,validation_split=0.15,callbacks=cb,verbose=0)
    P=m.predict(Xs[te],verbose=0); p=(P[:,0]>.5).astype(int) if n_out==1 else P.argmax(1)
    labs=[0,1] if n_out==1 else list(range(6))
    pr,rc,f1,sup=precision_recall_fscore_support(yy[te],p,labels=labs,zero_division=0)
    res=dict(cfg=cfg,n_train=int(len(tr)),n_test=int(len(te)),acc=float((p==yy[te]).mean()),
        cm=confusion_matrix(yy[te],p,labels=labs).tolist(),precision=pr.tolist(),recall=rc.tolist(),f1=f1.tolist(),support=sup.tolist(),
        epochs=len(h.history['loss']),best_val_loss=float(min(h.history['val_loss'])),seconds=round(time.time()-t0,1))
    if n_out==6:
        res['merged_acc']=float((np.where(p==5,4,p)==np.where(yy[te]==5,4,yy[te])).mean())
        res['test_regime_counts']=c.iloc[te].regime.value_counts().to_dict()
        # per-regime accuracy on the test rows
        res['acc_by_regime']={r:float((p[c.iloc[te].regime.values==r]==yy[te][c.iloc[te].regime.values==r]).mean()) for r in c.iloc[te].regime.unique()}
        # predictions on excluded rows that fall in the test partition (clearing-phase evaluation)
        if cfg['data']!='E0' and len(excl_te):
            pe=m.predict(Xs[excl_te],verbose=0).argmax(1); sub=c.iloc[excl_te]
            res['excluded_test_eval']=[dict(regime=r,label=CLASSES[yy[i]],phys=ph,pred=CLASSES[q]) for i,r,ph,q in zip(excl_te,sub.regime,sub.phys,pe)]
    json.dump(res,open(fn,'w'))
    print(f"{name}: acc={res['acc']:.4f} n_te={len(te)} ep={res['epochs']} {res['seconds']}s",flush=True)
def grid():
    G=[]
    for s in range(5):
        for part in ['random','head','mid','tail']:
            G.append(dict(task='class',data='E0',clip='train',scaler='minmax',part=part,seed=s))
            G.append(dict(task='class',data='E0',clip='global',scaler='minmax',part=part,seed=s))
            G.append(dict(task='class',data='E1',clip='train',scaler='minmax',part=part,seed=s))
            G.append(dict(task='class',data='E2',clip='train',scaler='minmax',part=part,seed=s))
        for part in ['random','mid','tail']:
            G.append(dict(task='class',data='E0',clip='train',scaler='robust',part=part,seed=s))
            G.append(dict(task='class',data='E0',clip='none',scaler='robust',part=part,seed=s))
        for part in ['mid','tail']: G.append(dict(task='class',data='E0',clip='train',scaler='minmax',part=part,seed=s,gap=100))
        G.append(dict(task='class',data='E0',clip='train',scaler='minmax',part='tail',seed=s,shuffle=0))
        G.append(dict(task='class',data='E0',clip='train',scaler='minmax',part='random',seed=s,drop_v0=1))
        for part in ['random','tail']: G.append(dict(task='detect',data='E0',clip='train',scaler='minmax',part=part,seed=s))
    return G
if __name__=='__main__':
    G=grid()
    if len(sys.argv)>1 and sys.argv[1]=='one': G=G[:1]
    print('runs in grid:',len(grid()),flush=True)
    for g in G: run(g)
