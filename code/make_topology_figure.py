"""Structure-only one-line diagram of the publisher's Simulink model (ANN_project.slx, parsed as XML).
Only structural facts and nameplate settings stored in the model are shown; no parameter is inferred."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':6.5})
fig,ax=plt.subplots(figsize=(3.45,1.35)); ax.set_xlim(0,10.05); ax.set_ylim(0,3.6); ax.axis('off')
y=1.9
def src(x,yy,lab):
    ax.add_patch(Circle((x,yy),0.28,fill=False,lw=0.8)); ax.text(x,yy,'~',ha='center',va='center',fontsize=8)
    ax.text(x,yy-0.5,lab,ha='center',va='top',fontsize=5.5)
def xfmr(x,lab):
    ax.add_patch(Circle((x-0.16,y),0.22,fill=False,lw=0.8)); ax.add_patch(Circle((x+0.16,y),0.22,fill=False,lw=0.8))
    ax.text(x,y+0.35,lab,ha='center',va='bottom',fontsize=5.5)
# left end: two sources on 11 kV bus
ax.plot([0.9,0.9],[1.0,2.8],'k',lw=1.4)  # bus L
src(0.35,2.45,''); src(0.35,1.35,'')
ax.plot([0.63,0.9],[2.45,2.45],'k',lw=.8); ax.plot([0.63,0.9],[1.35,1.35],'k',lw=.8)
ax.text(0.35,3.05,'2 × 11 kV',ha='center',fontsize=5.5); ax.text(0.35,0.55,'(+ load)',ha='center',fontsize=5.3)
ax.plot([0.9,1.55],[y,y],'k',lw=.8); xfmr(1.87,'11/132 kV\n100 MVA'); ax.plot([2.19,2.6],[y,y],'k',lw=.8)
# line sections and fault
ax.plot([2.6,2.6],[1.3,2.5],'k',lw=1.4); ax.plot([2.6,4.6],[y,y],'k',lw=.8); ax.text(3.6,y+0.12,'33 km',ha='center',va='bottom',fontsize=5.5)
ax.plot([4.6,4.6],[y,1.0],'k',lw=.8); ax.text(4.6,0.95,'F',ha='center',va='top',fontsize=7,fontweight='bold'); ax.text(4.6,0.55,'mid-line fault block\n(5 faults, 0.001 Ω)',ha='center',va='top',fontsize=5)
ax.plot([4.6,6.6],[y,y],'k',lw=.8); ax.text(5.6,y+0.12,'33 km',ha='center',va='bottom',fontsize=5.5); ax.plot([6.6,6.6],[1.3,2.5],'k',lw=1.4)
ax.text(4.6,3.2,'132 kV distributed-parameter line',ha='center',fontsize=5.5)
ax.plot([6.6,7.0],[y,y],'k',lw=.8); xfmr(7.32,'132/11 kV\n100 MVA'); ax.plot([7.64,8.05],[y,y],'k',lw=.8)
# measurement
ax.add_patch(Rectangle((8.05,y-0.22),0.5,0.44,fill=False,lw=0.8)); ax.text(8.3,y,'V–I',ha='center',va='center',fontsize=5.3)
ax.text(8.3,y-0.35,'measured\n(11 kV side)',ha='center',va='top',fontsize=5)
ax.plot([8.55,8.95],[y,y],'k',lw=.8); ax.plot([8.95,8.95],[1.3,2.5],'k',lw=1.4)
ax.plot([8.95,9.3],[2.2,2.2],'k',lw=.8); src(9.58,2.2,''); ax.text(9.45,2.85,'11 kV',ha='center',fontsize=5.5)
ax.text(9.5,1.35,'+ loads',ha='center',fontsize=5.3)
plt.tight_layout(pad=0.1); plt.savefig('figures/fig0_topology.png',dpi=300); print('ok')
