import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"font.family":"Arial","pdf.fonttype":42,"ps.fonttype":42,"mathtext.fontset":"custom","mathtext.rm":"Arial","mathtext.it":"Arial:italic","mathtext.bf":"Arial:bold","mathtext.cal":"Arial","font.size":8,"axes.linewidth":0.6,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b",
 "xtick.color":"#52514e","ytick.color":"#52514e","xtick.labelsize":7.5,"ytick.labelsize":7.5,"axes.grid":True,"grid.color":"#e6e5e1","grid.linewidth":0.5,
 "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"legend.fontsize":7.5,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight"})
C={"blue":"#2a78d6","orange":"#eb6834","aqua":"#1baf7a","yellow":"#eda100","violet":"#4a3aa7","ink":"#0b0b0b","muted":"#8a8984"}
N=json.load(open("ms_numbers.json")); D=N["datasets"]; E_R=N["constants"]["E_R"]; E_OP=N["constants"]["E_OP"]; E_S=N["constants"]["E_SENSE"]; MAC=N["constants"]["MACS"]
ORDER=["intel_time","machine_temp","iio_networkin","twitter_aapl","taxi","grok_asg_anomaly","cpu","rds_cpu_util","occupancy","exchange_cpc","traffic"]
LABEL={"intel_time":"Intel (temp.)","machine_temp":"machine_temp","iio_networkin":"iio_networkin","twitter_aapl":"twitter_aapl","taxi":"taxi","grok_asg_anomaly":"grok_asg",
       "cpu":"cpu","rds_cpu_util":"rds_cpu","occupancy":"occupancy","exchange_cpc":"exchange_cpc","traffic":"traffic"}
TAUS=["0.05","0.1","0.2","0.5"]

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
# ---- Fig 1: Intel operating points (two panels, one axis each) --------------
d=D["intel_time"]; taus=[float(t) for t in TAUS]
rho=[d["by_tau"][t]["test"]["persistence"]["rho"] for t in TAUS]; err=[d["by_tau"][t]["test"]["persistence"]["mean_err"] for t in TAUS]; mx=[d["by_tau"][t]["test"]["persistence"]["max_err"] for t in TAUS]
fig,ax=plt.subplots(1,2,figsize=(6.8,2.3))
ax[0].plot(taus,rho,"-o",color=C["blue"],lw=1.6,ms=4); ax[0].set_xscale("log"); ax[0].set_xlabel("tolerance $\\tau$ (°C)"); ax[0].set_ylabel("suppression $\\rho$ (test span)"); ax[0].set_ylim(0.8,1.02); ax[0].set_yticks([0.80,0.85,0.90,0.95,1.00])
for i,(x,y) in enumerate(zip(taus,rho)): ax[0].annotate("%.3f"%y,(x,y),textcoords="offset points",xytext=(6,-12) if i==0 else (0,6),ha="left" if i==0 else "center",fontsize=7,color=C["ink"])
ax[1].plot(taus,err,"-o",color=C["blue"],lw=1.6,ms=4,label="mean absolute error"); ax[1].plot(taus,mx,"--s",color=C["orange"],lw=1.2,ms=3.5,label="max error (= $\\tau$)")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("tolerance $\\tau$ (°C)"); ax[1].set_ylabel("reconstruction error at sink (°C)"); ax[1].legend(loc="upper left")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
for a in ax: a.xaxis.set_major_locator(FixedLocator(taus)); a.xaxis.set_major_formatter(FixedFormatter(["0.05","0.10","0.20","0.50"])); a.xaxis.set_minor_locator(NullLocator())
YT=[0.02,0.05,0.1,0.2,0.5]; ax[1].yaxis.set_major_locator(FixedLocator(YT)); ax[1].yaxis.set_major_formatter(FixedFormatter(["0.02","0.05","0.10","0.20","0.50"])); ax[1].yaxis.set_minor_locator(NullLocator())
fig.subplots_adjust(wspace=0.38)
for a,t in zip(ax,("(a)","(b)")): a.text(-0.18,1.02,t,transform=a.transAxes,fontsize=9,fontweight="bold")
fig.savefig("fig1_intel_operating_points.png"); save_panels(fig,"fig1_intel_operating_points",lambda f,n: f.savefig(n+".png")); plt.close(fig)
# ---- Fig 2: cross-dataset suppression at tau=0.10 (dot plot) ----------------
fig,ax=plt.subplots(figsize=(6.8,3.6)); ys=np.arange(len(ORDER))[::-1]
for y,ds in zip(ys,ORDER):
    b=D[ds]["by_tau"]["0.1"]; ref=b["test"][b["selected"]]["rho"]; per=b["test"]["persistence"]["rho"]
    ax.plot([per],[y],marker="|",ms=11,mew=1.6,color=C["muted"],ls="none")
    ax.plot([ref],[y],marker="D",ms=5,color=C["blue"],ls="none")
    for m,col,mk,off in (("lightweight",C["orange"],"o",0.16),("cnn_lstm",C["violet"],"^",-0.16)):
        e=D[ds]["learned"].get(m,{})
        if e.get("pending"): continue
        ax.errorbar([e["rho_mean"]],[y+off],xerr=[[e["rho_sd"]],[e["rho_sd"]]],fmt=mk,ms=4.2,color=col,ecolor=col,elinewidth=0.8,capsize=1.5)
ax.set_yticks(ys); ax.set_yticklabels([LABEL[d] for d in ORDER]); ax.set_xlim(0,1); ax.set_xlabel("test-span suppression $\\rho$ at $\\tau$ = 0.10")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker="|",ms=9,mew=1.6,color=C["muted"],ls="none",label="persistence"),Line2D([],[],marker="D",ms=5,color=C["blue"],ls="none",label="low-cost reference (selected on validation)"),
 Line2D([],[],marker="o",ms=4.2,color=C["orange"],ls="none",label="lightweight, 208 MACs (mean ± SD, 20 seeds)"),Line2D([],[],marker="^",ms=4.2,color=C["violet"],ls="none",label="CNN-LSTM, 99,488 MACs (mean ± SD, 20 seeds)")],
 loc="lower right",fontsize=7)
fig.savefig("fig2_cross_dataset.png"); plt.close(fig)
# ---- Fig 3: net saving per node-round vs radio cost (swept range) ------------
fig,ax=plt.subplots(figsize=(6.8,3.0)); er=np.logspace(-4,np.log10(3e-3),300)
sty={("taxi","cnn_lstm"):(C["violet"],"-"),("taxi","lightweight"):(C["orange"],"-"),("exchange_cpc","cnn_lstm"):(C["violet"],"--"),("exchange_cpc","lightweight"):(C["orange"],"--"),("intel_time","cnn_lstm"):(C["violet"],":"),("intel_time","lightweight"):(C["orange"],":")}
for (ds,m),(col,ls) in sty.items():
    e=D[ds]["learned"].get(m,{}); b=D[ds]["by_tau"]["0.1"]
    if e.get("pending"): continue
    dE=(MAC[m]-b["selected_macs"])*E_OP; g=er*e["delta"]-dE
    ax.plot(er,g*1e6,color=col,ls=ls,lw=1.5,marker="^" if m=="cnn_lstm" else "o",ms=3.4,markevery=(0.03 if m=="cnn_lstm" else 0.09,0.12),label="%s · %s"%(LABEL[ds],"CNN-LSTM" if m=="cnn_lstm" else "lightweight"))
    if e["crossover"] and 1e-4<=e["crossover"]<=3e-3:
        ax.plot([e["crossover"]],[0],marker="o",ms=6,mfc="white",mec=col,mew=1.4,ls="none",zorder=5)   # open circle: distinct from the filled lightweight markers
        lab=("$e_{\\mathrm{r}}$* = %.1f×10$^{-4}$ J"%(e["crossover"]*1e4)) if e["crossover"]<1e-3 else ("$e_{\\mathrm{r}}$* = %.1f×10$^{-3}$ J"%(e["crossover"]*1e3))
        if e["crossover"]<1e-3: ax.annotate(lab,(e["crossover"],0),textcoords="offset points",xytext=(-60,34),fontsize=7.5,color=col,ha="center",arrowprops=dict(arrowstyle="-",color=col,lw=0.6,shrinkA=1,shrinkB=3),bbox=dict(boxstyle="square,pad=0.15",fc="white",ec="none",alpha=0.9))
        else: ax.annotate(lab,(e["crossover"],0),textcoords="offset points",xytext=(4,-13 if m=="cnn_lstm" else 8),fontsize=7.5,color=col,bbox=dict(boxstyle="square,pad=0.15",fc="white",ec="none",alpha=0.9))
ax.axhline(0,color=C["ink"],lw=0.7); ax.axvline(E_R,color=C["muted"],lw=0.8,ls="-.")
y0,y1=ax.get_ylim(); ax.text(E_R*1.06,y0+0.03*(y1-y0),"Intel anchor\n3.8×10$^{-4}$ J",fontsize=7.5,color=C["muted"],va="bottom",ha="left",bbox=dict(boxstyle="square,pad=0.15",fc="white",ec="none",alpha=0.9))
ax.set_xscale("log"); ax.set_xlim(1e-4,3e-3); ax.set_xlabel("radio cost per report $e_{\\mathrm{r}}$ (J)"); ax.set_ylabel("net saving vs. low-cost reference\n(µJ per node-round)")
ax.legend(ncol=3,fontsize=7,loc="upper left",frameon=True,framealpha=1,facecolor="white",edgecolor="none")
fig.savefig("fig3_net_saving_vs_radio_cost.png"); plt.close(fig)
# ---- Fig 4: tolerance sweep heat map (diverging, gray midpoint) ---------------
have=[ds for ds in ORDER if "tau_sweep_lightweight" in D[ds]]
M=np.array([[D[ds]["tau_sweep_lightweight"][t]["delta"] for t in TAUS] for ds in have]); S=np.array([[D[ds]["tau_sweep_lightweight"][t]["sig"] for t in TAUS] for ds in have])
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
cmap=LinearSegmentedColormap.from_list("div",[C["orange"],"#d9d8d3",C["blue"]]); lim=float(np.abs(M).max())
fig,ax=plt.subplots(figsize=(4.2,3.6))
# vector cells (pcolormesh) instead of an embedded bitmap (imshow); same matrix, same orientation (first row at the top)
im=ax.pcolormesh(np.arange(5)-0.5,np.arange(len(have)+1)-0.5,M,cmap=cmap,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),edgecolors="face",linewidth=0.2)
ax.set_xlim(-0.5,3.5); ax.set_ylim(len(have)-0.5,-0.5)
ax.set_xticks(range(4)); ax.set_xticklabels(["0.05","0.10","0.20","0.50"]); ax.set_xlabel("tolerance $\\tau$"); ax.set_yticks(range(len(have))); ax.set_yticklabels([LABEL[d] for d in have]); ax.grid(False)
for i in range(len(have)):
    for j in range(4): ax.text(j,i,("%+.3f%s"%(M[i,j],"*" if S[i,j] else "")).replace("-","\u2212"),ha="center",va="center",fontsize=7,color=C["ink"])
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.03); cb.set_label("Δ$\\rho$ = lightweight − low-cost reference",fontsize=7); cb.ax.tick_params(labelsize=7); cb.solids.set_rasterized(False); cb.solids.set_edgecolor("face"); cb.solids.set_linewidth(0.6)   # vector colour bar; edges in the face colour close the hairline seams some viewers show
ax.set_title("* significant after Bonferroni",fontsize=7.5,loc="left",color=C["muted"])
fig.savefig("fig4_tolerance_sweep.png"); plt.close(fig)
print("figures written")
