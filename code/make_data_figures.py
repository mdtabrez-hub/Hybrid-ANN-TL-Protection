import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import pandas as pd, numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'legend.fontsize':6,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'lines.linewidth':0.7})
R=DATA; Ts=50e-6
d=pd.read_excel(R+'detect_dataset.xlsx'); reg=pd.read_csv('results/detect_row_regimes.csv')['regime'].values
t=np.arange(len(d))*Ts*1e3; I=d[['Ia','Ib','Ic']].values; I0=I.sum(1)/3; S=d['Output (S)'].values
ev=[('LG\n(A–G)',25,75),('LLG\n(AB–G)',125,175),('LL\n(B–C)',225,275),('LLL',325,375),('LLLG',425,475)]
fig,ax=plt.subplots(2,1,figsize=(7.16,2.75),sharex=True,gridspec_kw={'height_ratios':[2.2,1]})
cols=['#1f4e79','#c55a11','#548235']
for j in range(3): ax[0].plot(t,I[:,j],color=cols[j],label=['$I_a$','$I_b$','$I_c$'][j])
ax[0].plot(t,3*I0,'k',lw=0.9,label='$3I_0$')
for k,(n,a,b) in enumerate(ev):
    for A in ax: A.axvspan(a,b,color='0.88',zorder=0)
    ax[0].text((a+b)/2,960,n,ha='center',va='bottom',fontsize=6.5)
ax[0].set_ylim(-1000,1250); ax[0].set_ylabel('Current (A)'); ax[0].legend(ncol=2,loc='center',frameon=False,bbox_to_anchor=(0.9,0.28))
ax[0].axvspan(481.8,600,color='#e2efda',zorder=0); ax[0].text(540,960,'class "No fault"\n(481.8–600 ms)',ha='center',va='bottom',fontsize=6.5)
# regime strip
code={'healthy':0,'sustained':1,'clearing_sametype':2,'partial_interruption':3,'post_interruption':4}
cl={1:'#7f7f7f',2:'#5b9bd5',3:'#c00000',4:'#ffc000'}; nm={1:'labelled fault: sustained',2:'clearing, same topology',3:'partial interruption (type ≠ label)',4:'after last-pole interruption'}
r=np.array([code[x] for x in reg])
for v in (1,2,3,4):
    m=(r==v).astype(float); ax[1].fill_between(t,0,m,step='post',color=cl[v],lw=0,label=nm[v])
ax[1].fill_between(t,0,(S==0)*0.25,step='post',color='#a9d18e',lw=0,label='labelled no fault')
ax[1].set_yticks([]); ax[1].set_ylim(0,1.05); ax[1].set_xlabel('Time (ms)'); ax[1].legend(ncol=3,loc='upper center',frameon=False,bbox_to_anchor=(0.5,-0.55))
ax[1].set_xlim(0,600)
plt.tight_layout(h_pad=0.3); plt.savefig('figures/fig1_timeline.png',dpi=300); plt.close()
fig,ax=plt.subplots(2,1,figsize=(3.45,2.9),sharex=True,gridspec_kw={'height_ratios':[2.4,0.7]})
cols=['#1f4e79','#c55a11','#548235']
for j in range(3): ax[0].plot(t,I[:,j],color=cols[j],label=['$I_a$','$I_b$','$I_c$'][j])
ax[0].plot(t,3*I0,'k',lw=0.9,label='$3I_0$')
for k,(n,a,b) in enumerate(ev):
    for A in ax: A.axvspan(a,b,color='0.88',zorder=0)
    ax[0].text((a+b)/2,950,n,ha='center',va='bottom',fontsize=5.3)
ax[0].set_ylim(-1000,1250); ax[0].set_ylabel('Current (A)'); ax[0].legend(ncol=4,loc='upper center',frameon=False,bbox_to_anchor=(0.5,-0.02),fontsize=5.5,handlelength=1.2,columnspacing=0.8)
ax[0].axvspan(481.8,600,color='#e2efda',zorder=0); ax[0].text(540,950,'No\nfault',ha='center',va='bottom',fontsize=5.3)
# regime strip
code={'healthy':0,'sustained':1,'clearing_sametype':2,'partial_interruption':3,'post_interruption':4}
cl={1:'#7f7f7f',2:'#5b9bd5',3:'#c00000',4:'#ffc000'}; nm={1:'labelled fault: sustained',2:'clearing, same topology',3:'partial interruption (type ≠ label)',4:'after last-pole interruption'}
r=np.array([code[x] for x in reg])
for v in (1,2,3,4):
    m=(r==v).astype(float); ax[1].fill_between(t,0,m,step='post',color=cl[v],lw=0,label=nm[v])
ax[1].fill_between(t,0,(S==0)*0.25,step='post',color='#a9d18e',lw=0,label='labelled no fault')
ax[1].set_yticks([]); ax[1].set_ylim(0,1.05); ax[1].set_xlabel('Time (ms)'); ax[1].legend(ncol=2,loc='upper center',frameon=False,bbox_to_anchor=(0.5,-0.95),fontsize=5.3)
ax[1].set_xlim(0,600)
plt.tight_layout(h_pad=0.3); plt.savefig('figures/fig1_timeline_col.png',dpi=300); plt.close()

# ---- Fig 2: zero-sequence evidence
c=pd.read_pickle(WORK+'/class_full.pkl')
c['regime']=reg[c.t_idx.values]; c.loc[c.vec=='0000','regime']='healthy'
fig,ax=plt.subplots(1,2,figsize=(3.5,1.75),gridspec_kw={'width_ratios':[1.25,1]})
order=['0000','1001','0110','1011','0111','1111']; lab=['No\nfault','LG','LL','LLG','LLL','LLLG']
data=[c.loc[c.vec==v,'I0'].abs().values+1e-3 for v in order]
ax[0].boxplot(data,whis=(1,99),showfliers=False,widths=.55,medianprops=dict(color='k'))
ax[0].set_yscale('log'); ax[0].set_xticks(range(1,7)); ax[0].set_xticklabels(lab,fontsize=5.8); ax[0].set_ylabel('$|I_0|$ (A)'); ax[0].set_title('(a) $|I_0|$ by class')
for v,col,n in [('0111','#1f4e79','LLL'),('1111','#c00000','LLLG')]:
    g=c[c.vec==v]; tt=(g.t_idx.values*Ts*1e3)-(g.t_idx.values.min()*Ts*1e3)
    ax[1].plot(tt,g.I0.abs().values+1e-4,color=col,lw=0.8,label=n)
ax[1].set_yscale('log'); ax[1].set_ylim(1e-3,2e3); ax[1].axvline(49.85,color='0.4',ls='--',lw=.6); ax[1].text(48.5,30,'switch-off',fontsize=5.5,ha='right')
ax[1].set_xlabel('Time from label start (ms)',fontsize=6.3); ax[1].set_ylabel('$|I_0|$ (A)'); ax[1].set_title('(b) within the event'); ax[1].legend(frameon=False,loc='center',bbox_to_anchor=(0.42,0.72))
plt.tight_layout(w_pad=0.6); plt.savefig('figures/fig2_zero_sequence.png',dpi=300); plt.close(); print('ok')
