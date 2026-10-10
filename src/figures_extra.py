"""Figures 2, 6, 7 and 9 of the manuscript (Figures 3, 4, 5 and 8 come from figures.py).
Inputs: ms_numbers.json, learned_<ds>.json, intel_time.npy, intel_time_meta.json and
intel_time_rows.csv (written by prepare_intel.py). Each figure is saved as PNG (300 dpi)
and as vector PDF. Style and colours follow figures.py."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib import rcParams
from matplotlib.lines import Line2D
rcParams.update({"font.family":"Arial","pdf.fonttype":42,"ps.fonttype":42,"mathtext.fontset":"custom","mathtext.rm":"Arial","mathtext.it":"Arial:italic","mathtext.bf":"Arial:bold","mathtext.cal":"Arial","font.size":8,"axes.linewidth":0.6,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b",
 "xtick.color":"#52514e","ytick.color":"#52514e","xtick.labelsize":7.5,"ytick.labelsize":7.5,"axes.grid":True,"grid.color":"#e6e5e1","grid.linewidth":0.5,
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"legend.fontsize":7.5,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight"})
C={"blue":"#2a78d6","orange":"#eb6834","aqua":"#1baf7a","yellow":"#eda100","violet":"#4a3aa7","ink":"#0b0b0b","muted":"#8a8984"}
N=json.load(open("ms_numbers.json")); D=N["datasets"]; E_R=N["constants"]["E_R"]; E_OP=N["constants"]["E_OP"]; E_S=N["constants"]["E_SENSE"]; MAC=N["constants"]["MACS"]
ORDER=["intel_time","machine_temp","iio_networkin","twitter_aapl","taxi","grok_asg_anomaly","cpu","rds_cpu_util","occupancy","exchange_cpc","traffic"]
LABEL={"intel_time":"Intel (temp.)","machine_temp":"machine_temp","iio_networkin":"iio_networkin","twitter_aapl":"twitter_aapl","taxi":"taxi","grok_asg_anomaly":"grok_asg",
       "cpu":"cpu","rds_cpu_util":"rds_cpu","occupancy":"occupancy","exchange_cpc":"exchange_cpc","traffic":"traffic"}
MODELS=(("lightweight",C["orange"],0.16,"lightweight"),("cnn_lstm",C["violet"],-0.16,"CNN-LSTM"))
MK={"lightweight":"o","cnn_lstm":"^"}   # marker per model, so the models differ without colour
def save(fig,name): fig.savefig(name+".png"); fig.savefig(name+".pdf"); plt.close(fig)
def entries(ds,m): return [r for r in json.load(open(f"learned_{ds}.json"))["entries"] if r["predictor"]==m]
def ref(ds):
    b=D[ds]["by_tau"]["0.1"]; return b["test"][b["selected"]]["rho"], b["selected_macs"]
jit=np.random.default_rng(0)

# ---- one graph per file (journal, under "Do not submit:": "Multiple images or graphs combined into one file") ----
# Each panel of a two-panel figure is also written on its own (name_a, name_b). The panels are copies of
# the combined figure, and the assert checks that the copied data equal the combined figure's data.
import pickle
def _artist_data(a):
    """Everything that carries data or a number: lines, collections, patches (bars, spans), texts, images, limits, scales and ticks."""
    pt=lambda p: (p.get_path().vertices.tolist(), p.get_patch_transform().get_matrix().tolist())
    return ([l.get_xydata().tolist() for l in a.lines]+[c.get_offsets().tolist() for c in a.collections]
            +[np.asarray(c.get_array()).tolist() if c.get_array() is not None else None for c in a.collections]
            +[pt(p) for p in a.patches]+[(list(t.get_position()),t.get_text(),list(getattr(t,"xy",()))) for t in a.texts]
            +[np.asarray(i.get_array()).tolist() for i in a.images]
            +[list(a.get_xlim()),list(a.get_ylim()),a.get_xscale(),a.get_yscale(),list(a.get_xticks()),list(a.get_yticks())]
            +[[t.get_text() for t in a.get_xticklabels()],[t.get_text() for t in a.get_yticklabels()]])
def _same(x,y): return json.dumps(x,default=str)==json.dumps(y,default=str)   # NaN-safe comparison (NaN != NaN in lists)
def save_panels(fig,name,saver,fix=None):
    axes=[a for a in fig.axes]
    for k,a in enumerate(axes):
        f2=pickle.loads(pickle.dumps(fig)); keep=f2.axes[k]
        for o in list(f2.axes):
            if o is not keep: f2.delaxes(o)
        assert _same(_artist_data(keep),_artist_data(a)), "panel data differ from the combined figure"
        if fix: fix(f2,keep,k)
        saver(f2,"%s_%s"%(name,"ab"[k])); plt.close(f2)

# ---- Figure 2: the prepared Intel matrix in calendar time -------------------
M=np.load("intel_time.npy"); t=pd.to_datetime(pd.read_csv("intel_time_rows.csv")["bin_start"]); meta=json.load(open("intel_time_meta.json"))
hr=t.dt.floor("h"); bins=pd.Series(1,index=hr).groupby(level=0).sum(); hot=pd.Series((M>40).sum(1),index=hr).groupby(level=0).sum()
full=pd.date_range(hr.min(),hr.max(),freq="h"); bins=bins.reindex(full,fill_value=0); hot=hot.reindex(full,fill_value=0)
sp=meta["split_calendar"]; spans=[("train (50%)",sp["train"],C["aqua"]),("validation (20%)",sp["val"],C["yellow"]),("test (30%)",sp["test"],C["orange"])]
fig,ax=plt.subplots(2,1,figsize=(6.8,3.6),sharex=True,gridspec_kw={"height_ratios":[2.2,1]})
for a in ax:
    for lab,(s0,s1),col in spans: a.axvspan(pd.Timestamp(s0),pd.Timestamp(s1),color=col,alpha=0.12,lw=0)
w=1/24
ax[0].bar(bins.index,bins.values,width=w,align="edge",color=C["blue"],lw=0); ax[0].set_ylabel("complete 31-s bins\nper hour (max 117)")
for lab,(s0,s1),col in spans:
    mid=pd.Timestamp(s0)+(pd.Timestamp(s1)-pd.Timestamp(s0))/2; ax[0].text(mid,1.02,lab,transform=ax[0].get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
ax[1].bar(hot.index,hot.values,width=w,align="edge",color=C["orange"],lw=0); ax[1].set_ylabel("readings\n> 40 °C / h")
ax[1].xaxis.set_major_locator(mdates.DayLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax[1].set_xlabel("calendar time, 2004")
for a,l in zip(ax,("(a)","(b)")): a.text(-0.1,1.02,l,transform=a.transAxes,fontsize=9,fontweight="bold",va="bottom")
def fix2(f2,keep,k):   # each panel on its own: dates under (a), split labels over (b)
    if k==0:
        keep.tick_params(labelbottom=True); keep.xaxis.set_major_locator(mdates.DayLocator()); keep.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); keep.set_xlabel("calendar time, 2004")
    else:
        for lab,(s0,s1),col in spans:
            mid=pd.Timestamp(s0)+(pd.Timestamp(s1)-pd.Timestamp(s0))/2; keep.text(mid,1.02,lab,transform=keep.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
save_panels(fig,"fig_intel_matrix",save,fix=fix2); save(fig,"fig_intel_matrix")

# ---- Figure 6: the criterion per seed (Δρ_s and g_s) -------------------------
fig,ax=plt.subplots(1,2,figsize=(6.8,3.9),sharey=True); ys=np.arange(len(ORDER))[::-1]
for y,ds in zip(ys,ORDER):
    rr,rm=ref(ds)
    for m,col,off,_ in MODELS:
        v=np.array([r["test_rho"] for r in entries(ds,m)]); dd=v-rr; g=(E_R*dd-(MAC[m]-rm)*E_OP)*1e6
        yy=y+off+jit.uniform(-0.07,0.07,len(v))
        ax[0].scatter(dd,yy,s=7,color=col,alpha=0.75,lw=0,marker=MK[m],clip_on=False); ax[1].scatter(g,yy,s=7,color=col,alpha=0.75,lw=0,marker=MK[m],clip_on=False)
for a in ax: a.axvline(0,color=C["ink"],lw=0.8)
ax[0].set_yticks(ys); ax[0].set_yticklabels([LABEL[d] for d in ORDER])
ax[0].set_xlabel(r"Δ$\rho_s$ = $\rho_s$ − $\rho_{\mathrm{ref}}$  (suppression)"); ax[1].set_xscale("symlog",linthresh=20); ax[1].set_xlabel(r"$g_s$ at the anchor (µJ per node-round, symlog)")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
def _intminus(v,_): return ("%d"%v).replace("-","\u2212")   # named function: the figure must stay picklable for the panel files
lo,hi=ax[1].get_xlim(); TK=[t for t in (-1000,-300,-100,-30,0,30,100,300) if lo<=t<=hi]   # labelled ticks on both sides of zero
ax[1].xaxis.set_major_locator(FixedLocator(TK)); ax[1].xaxis.set_minor_locator(NullLocator()); ax[1].xaxis.set_major_formatter(FuncFormatter(_intminus))
for a,l in zip(ax,("(a)","(b)")): a.text(0,1.01,l,transform=a.transAxes,fontsize=9,fontweight="bold",va="bottom")
fig.legend(handles=[Line2D([],[],marker=MK[m],ls="none",color=c,ms=5,label=f"{lab}, 20 seeds") for m,c,_,lab in MODELS],loc="lower center",ncol=2,bbox_to_anchor=(0.5,-0.03))
fig.subplots_adjust(bottom=0.16)
def fix6(f2,keep,k):   # stream names on both panels; the shared legend goes under the single panel
    keep.tick_params(labelleft=True)
    for lg in list(f2.legends): lg.remove()
    keep.legend(handles=[Line2D([],[],marker=MK[m],ls="none",color=c,ms=5,label=f"{lab}, 20 seeds") for m,c,_,lab in MODELS],loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2)
save_panels(fig,"fig_per_seed",save,fix=fix6); save(fig,"fig_per_seed")

# ---- Figure 7: modelled energy per node-round for all five arms --------------
fig,ax=plt.subplots(figsize=(6.8,3.6)); ys=np.arange(len(ORDER))[::-1]
def energy(rho,mac): return (E_S+mac*E_OP+(1-rho)*E_R)*1e6
for y,ds in zip(ys,ORDER):
    b=D[ds]["by_tau"]["0.1"]; te=b["test"]
    ax.plot([energy(te["persistence"]["rho"],0)],[y],marker="|",ms=11,mew=1.6,color=C["muted"],ls="none")
    ax.plot([energy(te["expsm"]["rho"],12)],[y],marker="s",ms=5,color=C["aqua"],ls="none")
    ax.plot([energy(te["linear_extrap"]["rho"],b["K"])],[y],marker="D",ms=5,color=C["blue"],ls="none")
    ax.plot([energy(D[ds]["learned"]["lightweight"]["rho_mean"],MAC["lightweight"])],[y],marker="o",ms=5,color=C["orange"],ls="none")
    ax.plot([energy(D[ds]["learned"]["cnn_lstm"]["rho_mean"],MAC["cnn_lstm"])],[y],marker="^",ms=5.5,color=C["violet"],ls="none")
ax.set_yticks(ys); ax.set_yticklabels([LABEL[d] for d in ORDER])
ax.set_xlabel("modelled energy per node-round at $e_{\\mathrm{r}}$ = 3.8×10$^{-4}$ J, $\\tau$ = 0.10  (µJ; lower is better)")
H=[Line2D([],[],marker="|",ms=9,mew=1.6,color=C["muted"],ls="none",label="persistence (0 MACs)"),Line2D([],[],marker="s",ms=5,color=C["aqua"],ls="none",label="exp. smoothing $\\alpha$* (12 MACs)"),
   Line2D([],[],marker="D",ms=5,color=C["blue"],ls="none",label="linear extrapolation $K$* ($K$ MACs)"),Line2D([],[],marker="o",ms=5,color=C["orange"],ls="none",label="lightweight (208 MACs), seed mean"),
   Line2D([],[],marker="^",ms=5.5,color=C["violet"],ls="none",label="CNN-LSTM (99,488 MACs), seed mean")]
ax.legend(handles=H,loc="upper center",bbox_to_anchor=(0.5,-0.14),ncol=3,fontsize=7)
save(fig,"fig_energy_axis")

# ---- Figure 9: validation-based checkpoint selection -------------------------
fig,ax=plt.subplots(1,2,figsize=(6.8,3.2),gridspec_kw={"width_ratios":[1.15,1]}); ys=np.arange(len(ORDER))[::-1]
for y,ds in zip(ys,ORDER):
    for m,col,off,_ in MODELS:
        e=entries(ds,m); ep=np.array([r["selected_epoch"] for r in e],float)
        ax[0].scatter(ep+jit.uniform(-4,4,len(ep)),y+off+jit.uniform(-0.08,0.08,len(ep)),s=9,color=col,alpha=0.75,lw=0,marker=MK[m],clip_on=False)
ax[0].set_yticks(ys); ax[0].set_yticklabels([LABEL[d] for d in ORDER]); ax[0].set_xticks(range(25,201,25)); ax[0].set_xlabel("selected checkpoint (epoch), per seed")
for m,col,_,lab in MODELS:
    vv=[];tt=[]
    for ds in ORDER:
        for r in entries(ds,m): vv.append(r["val_rho"]); tt.append(r["test_rho"])
    ax[1].scatter(vv,tt,s=7,color=col,alpha=0.7,lw=0,label=lab,marker=MK[m],clip_on=False)
ax[1].plot([0,1],[0,1],color=C["ink"],lw=0.7); ax[1].set_xlim(0,1); ax[1].set_ylim(0,1)
ax[1].set_xlabel("validation $\\rho$ of the selected checkpoint"); ax[1].set_ylabel("test $\\rho$ of the same checkpoint"); ax[1].legend(loc="upper left",markerscale=2.2)
from matplotlib.lines import Line2D
ax[0].legend(handles=[Line2D([],[],marker=MK[m],ms=4,color=col,ls="none",label=lab) for m,col,_,lab in MODELS],loc="lower right",bbox_to_anchor=(1.0,1.0),ncol=2,borderaxespad=0.2,handletextpad=0.3,columnspacing=1.0)
for a,l in zip(ax,("(a)","(b)")): a.text(0,1.01,l,transform=a.transAxes,fontsize=9,fontweight="bold",va="bottom")
save_panels(fig,"fig_checkpoints",save); save(fig,"fig_checkpoints")
print("figures 2, 6, 7, 9 written (PNG + PDF)")

# ---- Section 6.9 statistics behind Figure 9 ----------------------------------
from scipy.stats import spearmanr
stats={}
for m,_,_,_ in MODELS:
    E=[r for ds in ORDER for r in entries(ds,m)]
    ep=np.array([r["selected_epoch"] for r in E]); v=np.array([r["val_rho"] for r in E]); t=np.array([r["test_rho"] for r in E])
    stats[m]=dict(runs=len(E),frac_at_epoch_cap=float(np.mean(ep==N["constants"]["EPOCH_CAP"])),frac_epoch_le_50=float(np.mean(ep<=50)),
                  spearman_val_test=float(spearmanr(v,t)[0]),median_abs_diff=float(np.median(np.abs(v-t))),max_abs_diff=float(np.max(np.abs(v-t))),
                  frac_test_below_val=float(np.mean(t<v)))
json.dump(stats,open("checkpoint_stats.json","w"),indent=1); print(json.dumps(stats,indent=1))
