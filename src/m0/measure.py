"""Build each predictor for Cortex-M0, run it in Nikov's Thumbulator (teamplay-D4.5),
read the per-inference event counters between the GPIOC markers, and convert them to
energy with the ten hardware-fitted models of Nikov et al. [21], Table II.
C1 executed instructions excluding MULS, C2 MULS, C3 taken branches, C4 RAM data reads,
C5 RAM writes, C6 flash data reads."""
import subprocess, re, json, sys, numpy as np
import os
BIN = os.environ.get("ARM_GCC_BIN", "xpack-arm-none-eabi-gcc-14.2.1-1.1/bin/"); GCC = BIN + "arm-none-eabi-gcc"; OC = BIN + "arm-none-eabi-objcopy"; SIZE = BIN + "arm-none-eabi-size"
SIM = os.environ.get("THUMB", "thumb/sim_main"); SIMP = os.environ.get("THUMBP", "thumbp/sim_main")
T2 = {"[20,OFF,0]": (0.964258, 1.652455, 2.091986, 1.109833, 0.650563, 0.633621),
      "[20,OFF,1]": (1.282474, 2.110668, 2.191545, 1.185609, 0.416602, 1.178991),
      "[20,ON,0]":  (1.003378, 1.885309, 1.802974, 1.122833, 0.849223, 0.475831),
      "[20,ON,1]":  (0.895879, 2.185851, 2.001178, 1.493364, 1.076354, 1.573758),
      "[24,OFF,0]": (0.959172, 1.888565, 1.357556, 1.089427, 0.993145, 0.562952),
      "[24,OFF,1]": (1.178558, 2.540429, 2.042475, 1.190892, 0.979651, 0.891088),
      "[24,ON,0]":  (0.985415, 1.933276, 1.448160, 1.075671, 1.011891, 0.617510),
      "[24,ON,1]":  (0.883755, 2.156046, 1.633465, 1.436556, 1.152560, 1.455166),
      "[48,OFF,1]": (1.096677, 2.364495, 1.627854, 1.173680, 0.681475, 0.652665),
      "[48,ON,1]":  (0.816331, 2.014612, 1.372157, 1.402116, 0.835035, 1.250446)}
def sh(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True)
def counters(binfile):
    err = sh(f"timeout 1200 {SIM} -s {binfile} 2>&1 >/dev/null").stdout
    blocks = re.split(r"TeamPlay: GPIOC\[0\] cleared[^\n]*\n", err)[1:]
    rows = []
    for b in blocks:
        g = lambda k: int(re.search(k + r":\s+(\d+)", b).group(1))
        m = re.search(r"\n16:\s+\d+, \[([^\]]*)\]", b); muls = int(m.group(1).split(",")[13])
        ins = g("Executed insns")
        rows.append((ins - muls, muls, g("Taken branches"), g("RAM data reads"), g("RAM writes"), g("Flash data reads")))
    return np.array(rows, float)
OPT = "-O2"
def build(src, hdr_cmd, out):
    r = sh(hdr_cmd); assert r.returncode == 0, r.stderr
    libm = "-lm" if "f32" in src else ""
    r = sh(f"{GCC} -mcpu=cortex-m0 -mthumb {OPT} -ffreestanding -specs=nano.specs -nostartfiles -T link.ld startup.c {src} {libm} -lc -lgcc -o {out}.elf && {OC} -O binary {out}.elf {out}.bin")
    assert r.returncode == 0, r.stderr
    size = sh(f"{SIZE} {out}.elf").stdout.split("\n")[1].split()[:3]
    return out + ".bin", dict(text=int(size[0]), data=int(size[1]), bss=int(size[2]))
def check(binfile, variant):
    out = sh(f"timeout 1200 {SIMP} -s {binfile} 2>/dev/null").stdout
    vals = [l for l in out.split("\n") if re.fullmatch(r"[0-9a-f]{8}", l.strip())]
    if variant == "f32":
        import struct
        c = np.array([struct.unpack(">f", bytes.fromhex(v))[0] for v in vals]); r = np.loadtxt("ref.txt")
        return float(np.abs(c - r).max())
    c = np.array([np.int32(np.uint32(int(v, 16))) for v in vals], np.int64); r = np.loadtxt("qref.txt", dtype=np.int64)
    return bool(np.array_equal(c, np.atleast_1d(r)))
if __name__ == "__main__":
    W = "../weights/"
    jobs = [("persistence", ""), ("expsm", ""), ("linear_extrap", ""),
            ("lightweight", W + "intel_time_lightweight_s0.npz"), ("cnn_lstm", W + "intel_time_cnn_lstm_s0.npz")]
    if len(sys.argv) > 1: jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    import os
    OPT = os.environ.get("OPT", "-O2"); globals()["OPT"] = OPT
    res = {}
    for kind, npz in jobs:
        for variant, src, gen in (("q", "model_q15.c", "qmodel.py"), ("f32", "model_f32.c", "gen.py")):
            if variant == "f32" and kind == "cnn_lstm" and "--skipf32" in sys.argv: continue
            b, size = build(src, f"python3 {gen} {kind} {npz}", f"{kind}_{variant}")
            ok = check(b, variant); C = counters(b); mean = C.mean(0)
            assert (ok is True) if variant == "q" else ok < 2e-6, (kind, variant, ok)   # outputs reproduced
            assert len(C) == 16, (kind, variant, len(C))                                  # one window per prediction
            E = {k: float(np.dot(w, mean)) for k, w in T2.items()}
            res[f"{kind}_{variant}"] = dict(check=ok, size=size, n=len(C), counters_mean=mean.tolist(),
                                           counters_min=C.min(0).tolist(), counters_max=C.max(0).tolist(),
                                           energy_nJ=E, energy_min_nJ=min(E.values()), energy_max_nJ=max(E.values()))
            print(f"{kind:14s} {variant:4s} check={ok} n={len(C)} C={np.round(mean).astype(int).tolist()} "
                  f"E={min(E.values())/1e3:.3f}-{max(E.values())/1e3:.3f} uJ size={size}", flush=True)
    json.dump(res, open(f"m0_measure{OPT.replace(' ','')}.json", "w"), indent=1)
