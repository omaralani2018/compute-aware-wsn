"""Re-checks the NumPy forward pass (fastfwd.py) against Keras on every saved model in weights/.
For each file the model is rebuilt with frozen_benchmark.build, the saved weights are loaded and
fastfwd.verify_against_keras's comparison is repeated (64 standard-normal inputs, rng seed 0),
recording the maximum absolute and relative error. Keys 'intel_' (the withdrawn epoch-indexed
matrix) are skipped. Writes keras_equivalence.json. Requires TensorFlow 2.21."""
import os, glob, json, re, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import frozen_benchmark as fb, fastfwd
rows=[]
for f in sorted(glob.glob("weights/*.npz")):
    name=os.path.basename(f)[:-4]; m_=re.match(r"(.+)_(lightweight|cnn_lstm)_s(\d+)$",name)
    ds,kind,seed=m_.group(1),m_.group(2),int(m_.group(3))
    if ds=="intel": continue
    z=np.load(f); W=[z[k] for k in sorted(z.files,key=lambda k:int(k.split("_")[1]))]
    m=fb.build(kind,seed); m.set_weights(W)
    rng=np.random.default_rng(0); L=m.input_shape[1]; Z=rng.standard_normal((64,L)).astype("float32")
    ref=np.asarray(m(Z[...,None] if kind=="cnn_lstm" else Z,training=False)).reshape(-1).astype(np.float64)
    got=fastfwd.forward(fastfwd.extract(m,kind),Z.astype(np.float64))
    err=float(np.abs(ref-got).max()); rel=err/max(1e-8,float(np.abs(ref).max()))
    rows.append(dict(dataset=ds,model=kind,seed=seed,max_abs_err=err,max_rel_err=rel))
rel=np.array([r["max_rel_err"] for r in rows])
out=dict(n_models=len(rows),max_rel_err=float(rel.max()),median_rel_err=float(np.median(rel)),tolerance=2e-4,all_within_tolerance=bool((rel<=2e-4).all()),models=rows)
json.dump(out,open("keras_equivalence.json","w"),indent=1)
print({k:v for k,v in out.items() if k!="models"})
