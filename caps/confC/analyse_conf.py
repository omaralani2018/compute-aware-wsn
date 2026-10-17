"""Pre-registered analysis for PREREG10.txt (written before any result of the six streams was inspected)."""
import os, sys, json, statistics as st
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../adaptA"))); sys.path.insert(0, HERE)
import numpy as np
import oracle_switch as O, conf_data as cd
fb = O.fb; ER, EOP, ES, LB, TAU = fb.E_R, fb.E_OP, fb.E_SENSE, fb.LOOK_BACK, fb.TAU
MAC = {"lightweight": 208, "cnn_lstm": 99488}
def en(rho, m): return ES + m * EOP + (1 - rho) * ER
def dps(M, f, u): return fb.dps(M, f, tau=TAU, look_back=LB, train_frac=1e-6, unit=u)

def cheap_select(VAL, TEST, u):                                  # identical rule to ms_numbers.cheap_select
    a = max(fb.ALPHA_GRID, key=lambda a: (dps(VAL, fb.make_expsm(a), u)["rho"], -a))
    k = max(fb.LIN_K_GRID, key=lambda k: (dps(VAL, fb.make_ar(k), u)["rho"], -k))
    arms = {"persistence": (None, 0), "expsm": (fb.make_expsm(a), 12), "linear_extrap": (fb.make_ar(k), k)}
    val = {n: dps(VAL, f, u)["rho"] for n, (f, _) in arms.items()}
    sel = max(val, key=val.get)
    test = {n: dps(TEST, f, u)["rho"] for n, (f, _) in arms.items()}
    return arms, val, test, sel

res = {}; tot = {k: 0.0 for k in ["ref", "lw", "cnn", "caps", "hind"]}
for ds in cd.FILES:
    try: A, u = cd.load(ds)
    except ValueError as e: res[ds] = {"evaluable": False, "why": str(e)}; continue
    T = len(A); tr, va = int(fb.SPLIT[0] * T), int(fb.SPLIT[1] * T)
    VAL, TEST = A[tr - LB:va], A[va - LB:]
    arms, val, test, sel = cheap_select(VAL, TEST, u); fR, mR = arms[sel]; vR, tR = val[sel], test[sel]
    L = json.load(open(os.path.join(HERE, f"learned_{ds}.json")))["entries"]
    by = {k: sorted([e for e in L if e["predictor"] == k], key=lambda e: e["seed"]) for k in MAC}
    assert all(len(v) == 20 for v in by.values()), (ds, {k: len(v) for k, v in by.items()})
    scr = {k: (MAC[k] - mR) * EOP >= (1 - vR) * ER for k in MAC}                       # Step 1
    net = {k: [(e["test_rho"] - tR) * ER - (MAC[k] - mR) * EOP for e in by[k]] for k in MAC}
    c1_false = [k for k in MAC if scr[k] and st.median(net[k]) > 0]
    caps_e, picks = [], {"ref": 0, "lightweight": 0, "cnn_lstm": 0}
    for i in range(20):                                                                  # Step 2
        opts = [("ref", 0.0, en(tR, mR))] + [(k, (by[k][i]["val_rho"] - vR) * ER - (MAC[k] - mR) * EOP, en(by[k][i]["test_rho"], MAC[k])) for k in MAC if not scr[k]]
        b = max(opts, key=lambda o: o[1]); picks[b[0]] += 1; caps_e.append(b[2])
    head = {}
    for span, X in (("val", VAL), ("test", TEST)):                                       # Step 3 (lightweight)
        n = (X.shape[0] - LB) * X.shape[1]; txR = O.static(X, fR, u).sum(); E_R = txR * ER + n * mR * EOP; G = []
        for i in range(20):
            fL = O.learned_arm(A, "lightweight", os.path.join(HERE, "weights", f"{ds}_lightweight_s{i}.npz"))
            txL = O.static(X, fL, u).sum(); E_L = txL * ER + n * 208 * EOP
            if span == "test": assert abs((1 - txL / n) - by["lightweight"][i]["test_rho"]) < 1e-9
            G.append(1 - O.oracle(X, [fR, fL], [mR, 208], 96, u)[0] / min(E_R, E_L))
        head[span] = float(np.median(G))
    e = dict(ref=en(tR, mR), lw=st.median(en(x["test_rho"], 208) for x in by["lightweight"]),
             cnn=st.median(en(x["test_rho"], 99488) for x in by["cnn_lstm"]), caps=st.median(caps_e))
    e["hind"] = min(e["ref"], e["lw"], e["cnn"])
    for k in tot: tot[k] += e[k]
    res[ds] = dict(evaluable=True, ref=sel, rho_ref_val=vR, rho_ref_test=tR, screened=scr, false_exclusions=c1_false,
                   median_net={k: st.median(v) for k, v in net.items()}, caps_picks=picks, headroom=head,
                   flag_val=head["val"] >= 0.05, flag_test=head["test"] >= 0.05, energy=e,
                   verified=all(x["receiver_verified"] for k in MAC for x in by[k]))
    r = res[ds]
    print("%-20s ref=%-13s rhoR val %.3f test %.3f | screened %s | false excl %s | picks %s | headroom val %.1f%% test %.1f%%" % (
        ds, sel, vR, tR, {k: v for k, v in scr.items() if v}, c1_false, picks, 100 * head["val"], 100 * head["test"]), flush=True)
ev = [d for d in res if res[d]["evaluable"]]
C1 = all(not res[d]["false_exclusions"] for d in ev)
agree = sum(res[d]["flag_val"] == res[d]["flag_test"] for d in ev); C2 = agree == len(ev)
ceil = 1 - tot["hind"] / tot["ref"]; C3 = ceil < 0.05; C4 = tot["caps"] < tot["cnn"]
print("evaluable %d/2 | C1 no false exclusion: %s | C2 flag agreement %d/%d: %s | C3 hindsight ceiling %.1f%%: %s | C4 CAPS %.1f%% below always-CNN: %s" % (
    len(ev), C1, agree, len(ev), C2, 100 * ceil, C3, 100 * (1 - tot["caps"] / tot["cnn"]), C4))
print("totals vs ref: lw %+.1f%%, cnn %+.1f%%, caps %+.1f%%" % tuple(100 * (tot[k] / tot["ref"] - 1) for k in ("lw", "cnn", "caps")))
json.dump(dict(streams=res, totals=tot, C1=C1, C2=C2, agree=agree, ceiling=ceil, C3=C3, C4=C4), open(os.path.join(HERE, "conf_summary.json"), "w"), indent=1, default=float)
