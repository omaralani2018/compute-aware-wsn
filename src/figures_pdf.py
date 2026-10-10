"""Runs figures.py and additionally writes every figure as vector PDF
(the journal asks for line art at >= 1000 dpi or in vector form)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.figure as mfig
_save=mfig.Figure.savefig
def _both(self,fname,*a,**k):
    _save(self,fname,*a,**k)
    if str(fname).endswith(".png"): _save(self,str(fname)[:-4]+".pdf",*a,**k)
mfig.Figure.savefig=_both
exec(compile(open("figures.py").read(),"figures.py","exec"))
