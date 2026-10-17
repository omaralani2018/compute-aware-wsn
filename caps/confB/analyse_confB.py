"""Pre-registered analysis for PREREG7 + Amendment 7a (written before any run of the four deployments)."""
import os, sys, json, pickle, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "code"))
import numpy as np
import frozen_benchmark as fb, fastfwd, segeval as se
ER, EOP, ES, LB, TAU, B, HMIN = fb.E_R, fb.E_OP, fb.E_SENSE, fb.LOOK_BACK, 0.10, 96, 0.05
MAC = {"lightweight": 208, "cnn_lstm": 99488}
def en(rho, m): return ES + m * EOP + (1 - rho) * ER

def forecaster(D, wfile):
    Xs, Ys = [], []
    for s in D["train"]:
        x = s["values"]
        for t in range(LB, len(x)): Xs.append(x[t - LB:t]); Ys.append(x[t])
    Xs, Ys = np.array(Xs), np.array(Ys); rng = np.random.default_rng(0)
    if len(Xs) > fb.MAX_WINDOWS:
        i = rng.choice(len(Xs), fb.MAX_WINDOWS, replace=False); Xs, Ys = Xs[i], Ys[i]
    mx, sx = Xs.mean(0), Xs.std(0) + 1e-8; my, sy = float(Ys.mean()), float(Ys.std() + 1e-8)
    w = np.load(wfile); P = ("lightweight", *[w["arr_%d" % i] for i in range(len(w.files))])
    return lambda W: fastfwd.forward(P, (np.asarray(W, float) - mx) / sx) * sy + my

def fc(f, h): return h[-1] if f is None else float(np.asarray(f(h[None, :])).reshape(-1)[0])

def sim(x, r, f, a, b):
    tx = 0
    for t in range(a, b):
        p = fc(f, r[t - LB:t])
        if abs(x[t] - p) <= TAU: r[t] = p
        else: r[t] = x[t]; tx += 1
    return tx

def seg_oracle(segs, arms, macs):
    """Greedy block oracle per segment; arm carries over between consecutive segments of a station."""
    cur, E = {}, 0.0
    for s in sorted(segs, key=lambda s: (s["mote"], s["bins"][0])):
        x = np.asarray(s["values"], float)
        if len(x) <= LB: continue
        r = x.copy(); c = cur.get(s["mote"], -1)
        for a in range(LB, len(x), B):
            b = min(a + B, len(x)); best = None
            for k, f in enumerate(arms):
                rr = r.copy(); tx = sim(x, rr, f, a, b)
                e = tx * ER + (b - a) * macs[k] * EOP + (ER if (c >= 0 and c != k) else 0.0)
                if best is None or e < best[0]: best = (e, k, rr)
            E += best[0]; c = best[1]; r = best[2]
        cur[s["mote"]] = c
    return E

res = {}; tot = {k: 0.0 for k in ["ref", "lw", "cnn", "caps", "hind"]}
for dep in ["fishnet", "genepi", "pdg2008", "plaine_morte"]:
    d = os.path.join(HERE, "run_" + dep)
    meta = json.load(open(os.path.join(d, "stb_meta.json")))
    if not meta["evaluable"]["obs"]: res[dep] = {"evaluable": False}; continue
    S = pickle.load(open(os.path.join(d, "stb_segments_obs.pkl"), "rb"))
    D = {sp: [s for s in S if s["span"] == sp] for sp in ("train", "val", "test")}
    ch = json.load(open(os.path.join(d, "stb_obs_cheap.json"))); sel = ch["selected"]; mR = ch["selected_macs"]
    vR, tR = ch["val"][sel], ch["test"][sel]["rho"]
    fR = None if sel == "persistence" else fb.make_expsm(ch["alpha"]) if sel == "expsm" else fb.make_ar(ch["K"])
    by = {}
    for k in MAC:
        rows = [json.load(open(os.path.join(d, "runs", f"stb_obs_{k}_s{i}.json"))) for i in range(20)]
        by[k] = [r for r in rows if r["status"] == "ok"]
    scr = {k: (MAC[k] - mR) * EOP >= (1 - vR) * ER for k in MAC}
    net = {k: [(e["test_rho"] - tR) * ER - (MAC[k] - mR) * EOP for e in by[k]] for k in MAC}
    false_excl = [k for k in MAC if scr[k] and st.median(net[k]) > 0]
    seeds = sorted(set(e["seed"] for e in by["lightweight"]) & set(e["seed"] for e in by["cnn_lstm"]))
    L = {k: {e["seed"]: e for e in by[k]} for k in MAC}
    caps_e, picks = [], {"ref": 0, "lightweight": 0, "cnn_lstm": 0}
    for i in seeds:
        opts = [("ref", 0.0, en(tR, mR))] + [(k, (L[k][i]["val_rho"] - vR) * ER - (MAC[k] - mR) * EOP, en(L[k][i]["test_rho"], MAC[k])) for k in MAC if not scr[k]]
        b = max(opts, key=lambda o: o[1]); picks[b[0]] += 1; caps_e.append(b[2])
    head = {}
    for sp in ("val", "test"):
        segs = D[sp]; V = [s["values"] for s in segs]
        rR = se.seg_dps(V, fR, TAU); nst = rR["n_steps"]; E_R = rR["n_tx"] * ER + nst * mR * EOP
        if sp == "val": assert abs(rR["rho"] - vR) < 1e-12
        G = []
        for e in by["lightweight"]:
            fL = forecaster(D, os.path.join(d, "weights_stb", f"obs_lightweight_s{e['seed']}.npz"))
            rL = se.seg_dps(V, fL, TAU); E_L = rL["n_tx"] * ER + nst * 208 * EOP
            if sp == "test": assert abs(rL["rho"] - e["test_rho"]) < 1e-9, (dep, e["seed"], rL["rho"], e["test_rho"])
            G.append(1 - seg_oracle(segs, [fR, fL], [mR, 208]) / min(E_R, E_L))
        head[sp] = float(np.median(G))
    e = dict(ref=en(tR, mR), lw=st.median(en(x["test_rho"], 208) for x in by["lightweight"]),
             cnn=st.median(en(x["test_rho"], 99488) for x in by["cnn_lstm"]), caps=st.median(caps_e))
    e["hind"] = min(e["ref"], e["lw"], e["cnn"])
    for k in tot: tot[k] += e[k]
    res[dep] = dict(evaluable=True, ref=sel, rho_ref_val=vR, rho_ref_test=tR, seeds_ok={k: len(by[k]) for k in MAC},
                    screened=scr, false_exclusions=false_excl, median_net={k: st.median(v) for k, v in net.items()},
                    caps_picks=picks, headroom=head, flag_val=head["val"] >= HMIN, flag_test=head["test"] >= HMIN, energy=e,
                    verified=all(x["receiver_verified"] for k in MAC for x in by[k]))
    r = res[dep]
    print("%-13s ref=%-13s rhoR val %.3f test %.3f | seeds %s | screened %s | false excl %s | picks %s | headroom val %.1f%% test %.1f%%" % (
        dep, sel, vR, tR, r["seeds_ok"], {k: v for k, v in scr.items() if v}, false_excl, picks, 100 * head["val"], 100 * head["test"]), flush=True)
ev = [d for d in res if res[d]["evaluable"]]
C1 = all(not res[d]["false_exclusions"] for d in ev)
agree = sum(res[d]["flag_val"] == res[d]["flag_test"] for d in ev); C2 = agree >= len(ev) - 1
ceil = 1 - tot["hind"] / tot["ref"]; C3 = ceil < 0.05; C4 = tot["caps"] < tot["cnn"]
print("evaluable %d/4 | C1 no false exclusion: %s | C2 flag agreement %d/%d: %s | C3 hindsight ceiling %.1f%%: %s | C4 CAPS %.1f%% below always-CNN: %s" % (
    len(ev), C1, agree, len(ev), C2, 100 * ceil, C3, 100 * (1 - tot["caps"] / tot["cnn"]), C4))
print("totals vs ref: lw %+.1f%%, cnn %+.1f%%, caps %+.1f%%" % tuple(100 * (tot[k] / tot["ref"] - 1) for k in ("lw", "cnn", "caps")))
json.dump(dict(deployments=res, totals=tot, C1=C1, C2=C2, agree=agree, ceiling=ceil, C3=C3, C4=C4), open(os.path.join(HERE, "confB_summary.json"), "w"), indent=1, default=float)
