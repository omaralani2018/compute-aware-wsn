"""Version 1.4. Figure 1 of the paper (methodology diagram): (a) one round of closed-loop dual prediction,
(b) the evaluation protocol. A drawing, not generated from data. Needs matplotlib and the Arial fonts.
    python3 fig_methodology.py      -> fig_methodology.png/.pdf (both panels) and fig_methodology_a, fig_methodology_b
Drawing units are pixels at 300 dpi; panel (a) uses y = 0..840 and panel (b) y = 845..2060."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"Arial","pdf.fonttype":42,"ps.fonttype":42,"mathtext.fontset":"custom","mathtext.rm":"Arial","mathtext.it":"Arial:italic","mathtext.bf":"Arial:bold","mathtext.cal":"Arial"})
W=2018
BLUE="#EEF2F9"; PEACH="#FDF0E8"; GREEN="#EDF6F0"; GOLDF="#FDF1D6"; GOLDE="#E8A317"
EDGE="#4A4A4A"; GREY="#8A8A8A"; ORANGE="#E0643A"; GOLDT="#9A5B00"
FS=7.6
ER=r"$e_{\mathrm{r}}$"   # radio energy per report (descriptive subscript, upright)

def make(y0,y1,x1=W):
    """A canvas showing the drawing coordinates 0..x1, y0..y1 (1 unit = 1 px at 300 dpi)."""
    fig=plt.figure(figsize=(x1/300,(y1-y0)/300),dpi=300)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,x1); ax.set_ylim(y1,y0); ax.axis("off")
    return fig,ax

def helpers(ax):
    def box(x0,y0,x1,y1,fc,ec=EDGE,lw=1.0):
        ax.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0,rounding_size=6",fc=fc,ec=ec,lw=lw))
    def txt(x,y,s,fs=FS,**k):
        k.setdefault("ha","center"); k.setdefault("va","center")
        ax.text(x,y,s,fontsize=fs,linespacing=1.25,**k)
    def arrow(p,q,color=EDGE,lw=1.2,cs="arc3",ms=9):
        ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=ms,color=color,lw=lw,connectionstyle=cs,shrinkA=0,shrinkB=0))
    return box,txt,arrow

def draw_a(ax):   # drawing coordinates y = 0..830
    box,txt,arrow=helpers(ax)
    txt(38,45,"(a)",fs=8.6,weight="bold",ha="left")   # title only in the caption (journal)
    box(60,150,402,370,BLUE); txt(231,260,"Sensor reads $x(t)$\n($E_{\\mathrm{sense}}$)")
    arrow((404,260),(483,260))
    box(487,130,965,390,PEACH)   # widened 15 units each side: text margin about 1.8 mm
    txt(726,195,"Predictor on the shared\nreconstructed state\n$\\hat{x}(t)$ = $f$($\\tilde{x}$($t$−12 … $t$−1))")
    ax.text(726,292,"$E_{\\mathrm{inf}}$ = $M$·$E_{\\mathrm{op}}$",fontsize=FS+0.6,color=GOLDT,ha="center",va="center",
            bbox=dict(boxstyle="round,pad=0.35",fc=GOLDF,ec=GOLDE,lw=1.2))
    txt(726,357,"(same predictor at node and sink)",fs=FS-0.6,color="#555555",style="italic")
    arrow((967,260),(1046,260))
    box(1050,150,1393,370,BLUE); txt(1221,260,"|$x(t)$ − $\\hat{x}(t)$| > $\\tau$ ?")
    box(1475,95,1765,235,GREEN); txt(1620,165,"report $x(t)$\n(radio cost "+ER+")")
    box(1475,285,1765,425,GREEN); txt(1620,355,"no report\n(zero report cost)")
    arrow((1395,215),(1471,165))
    arrow((1395,305),(1471,355))
    txt(1440,150,"yes",fs=FS,color="#444444")
    txt(1440,372,"no",fs=FS,color="#444444")
    ax.plot([1768,1797,1797],[165,165,462],color=GREY,lw=1.1)
    ax.plot([1768,1797],[355,355],color=GREY,lw=1.1)
    ax.plot([452,1797],[462,462],color=GREY,lw=1.1)
    arrow((452,462),(452,500),color=GREY,lw=1.1,ms=8)
    arrow((1372,462),(1372,500),color=GREY,lw=1.1,ms=8)
    txt(1335,443,"either outcome updates both states",fs=FS-0.6,color=GREY)
    box(60,505,862,690,PEACH); txt(461,597,"Node updates locally:\n$\\tilde{x}(t)$ = $x(t)$ if reported, else $\\hat{x}(t)$")
    box(980,505,1782,690,PEACH); txt(1381,597,"Sink updates locally:\n$\\tilde{x}(t)$ = received $x(t)$, else its own $\\hat{x}(t)$")
    ax.add_patch(FancyArrowPatch((190,503),(535,392),arrowstyle="-|>",mutation_scale=11,color=ORANGE,lw=1.6,connectionstyle="arc3,rad=-0.15",shrinkA=0,shrinkB=0))
    txt(72,415,"next input",fs=FS,color=ORANGE,ha="left")
    txt(921,733,"identical reconstruction on both sides under synchronised, lossless operation (no state is transmitted)",fs=FS-0.6,color=GREY)
    txt(62,795,"Per-round energy = $E_{\\mathrm{sense}}$ + $E_{\\mathrm{inf}}$ + "+ER+"·[reported].  $\\rho$ = fraction of rounds without a report; sink error ≤ $\\tau$ after initialisation.",fs=FS,ha="left")

def draw_b(ax):   # drawing coordinates y = 845..2060
    box,txt,arrow=helpers(ax)
    txt(38,880,"(b)",fs=8.6,weight="bold",ha="left")   # title only in the caption (journal)
    bx=[(40,460),(515,935),(990,1410),(1465,1885)]   # wider boxes: the text keeps a margin at the full font size
    lab=["1. Chronological split\n50% train · 20% validation\n· 30% test",
         "2. Train learned predictors\non training data only\n(20 seeds)",
         "3. Select on validation only\n$\\alpha$, $K$, checkpoint,\nlow-cost reference",
         "4. Freeze all choices"]
    fcs=[BLUE,BLUE,BLUE,GREEN]
    for (a,b),l,fc in zip(bx,lab,fcs):
        box(a,950,b,1150,fc); txt((a+b)/2,1050,l)
    for i in range(3): arrow((bx[i][1]+2,1050),(bx[i+1][0]-4,1050))
    arrow((1530,1152),(1530,1266))
    box(286,1270,1638,1460,GREEN)   # boxes 5 and 6 centred under the step row (x = 962)
    txt(962,1365,"5. Closed-loop test, scored once (panel a)\nsuppression $\\rho$, sink error; an independent receiver rebuilds the series from the reports alone")
    arrow((962,1462),(962,1540))
    box(40,1545,1885,2020,GOLDF,ec=GOLDE,lw=2.4)   # same width as the step row
    txt(962,1622,"6. Energy decision (the criterion)",fs=8.4,weight="bold")
    txt(962,1735,"Net modelled saving  $g$ = "+ER+"·Δ$\\rho$  −  ($E_{\\mathrm{inf}}$ − $E_{\\mathrm{inf{,}ref}}$)",fs=8.4)
    txt(962,1815,"communication saved by extra suppression  −  extra computation",fs=FS,color=GREY)
    txt(962,1885,"$g$ > 0: the predictor saves energy     $g$ < 0: it costs energy     crossover "+ER+"* where $g$ = 0",fs=FS+0.2)
    txt(962,1960,"Per-seed net savings are tested against zero across 20 seeds (Wilcoxon signed-rank, Bonferroni-corrected)",fs=FS,color="#333333",style="italic")

def save(fig,name):
    kw=dict(facecolor="white")
    fig.savefig(name+".png",dpi=300,**kw); fig.savefig(name+".pdf",**kw); plt.close(fig)

# combined figure (embedded in the manuscript) and one file per panel (journal: one graph per file)
# canvas width = content width + a right margin equal to the left one (about 3 mm), so no blank strip is left
fig,ax=make(0,2060,1923); draw_a(ax); draw_b(ax); save(fig,"fig_methodology")
fig,ax=make(0,840,1835);  draw_a(ax); save(fig,"fig_methodology_a")
fig,ax=make(845,2060,1923); draw_b(ax); save(fig,"fig_methodology_b")
