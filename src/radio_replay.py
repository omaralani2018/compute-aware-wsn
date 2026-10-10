"""Replay of the saved Intel suppression traces (published preparation V0, test span) through the network
model of routing_baselines.py (first-order radio, 4,000-bit packets, the 30-mote layout, the
sink outside the field corner), to compare the energy a suppressed report really saves with the
average all-report cost e_r = 3.8e-4 J used by the criterion.

For every round (test row) and every node, the trace says whether the node sent. Scenarios, with the
same topology and the same per-round cluster-head draws for every predictor:
  direct      each sent report goes straight to the sink: the sender pays its transmission.
  leach_agg   LEACH with p = 0.2 (6 cluster heads per round, the best p of the routing comparison, routing_30motes.json), redrawn every round:
              a member's sent report costs its transmission to the nearest head and the head's
              reception; each head sends one aggregated packet to the sink if it or any member sent.
  leach_fwd   as leach_agg, but the head forwards one packet per report sent by itself or a member.
  criterion   e_r per sent report (the paper's model).
Energy per scored node-round: sensing + inference (MACs x 1 nJ) + radio. Initialisation rows are sent by
every predictor alike and are included. Writes radio_replay.json: per seed, the net saving and radio difference
against persistence; "mean" averages per-seed values (the per-extra-report cost as a mean of per-seed ratios);
"pooled_over_seeds" divides the summed radio difference by the summed extra reports (the effective costs
quoted in Section 6.9); "seeds_with_positive_net_saving" lists the seeds that save energy."""
import json, numpy as np, pandas as pd
E_ELEC, EPS_FS, EPS_MP, BITS, E_SENSE, E_OP, E_R = 50e-9, 10e-12, 0.0013e-12, 4000, 1e-7, 1e-9, 3.8e-4
D0 = np.sqrt(EPS_FS / EPS_MP); P_CH = 0.2
def etx(d): return BITS * E_ELEC + np.where(d < D0, BITS * EPS_FS * d ** 2, BITS * EPS_MP * d ** 4)
ERX = BITS * E_ELEC
meta = json.load(open("intel_time_meta.json")); motes = meta["motes"]
locs = pd.read_csv("mote_locs.txt", sep=r"\s+", names=["moteid", "x", "y"]).set_index("moteid")
pos = locs.loc[motes, ["x", "y"]].to_numpy(float); N = len(pos)
span = float(np.linalg.norm(pos.max(0) - pos.min(0))); sink = pos.min(0) - 0.15 * span
d_sink = np.linalg.norm(pos - sink, axis=1); dmat = np.linalg.norm(pos[:, None] - pos[None], axis=2)
I = np.load("cells/v0_index.npz"); R = len(I["mote"]) // N
def sent(name): return np.load(f"cells/v0_{name}.npz")["sent"].reshape(N, R).T          # rounds x nodes
rng = np.random.default_rng(0); nCH = max(1, int(round(P_CH * N)))
CH = np.array([rng.choice(N, nCH, replace=False) for _ in range(R)])                    # same draws for every predictor
isCH = np.zeros((R, N), bool); isCH[np.arange(R)[:, None], CH] = True
head = np.array([[CH[r][np.argmin(dmat[i, CH[r]])] if not isCH[r, i] else i for i in range(N)] for r in range(R)])
d_head = dmat[np.arange(N)[None, :], head]

def radio(S):
    out = {}
    out["direct"] = float((S * etx(d_sink)[None, :]).sum())
    member = S & ~isCH
    e_member = (member * (etx(d_head) + ERX)).sum()                                         # member tx + head rx
    agg = np.zeros((R, N), bool)
    np.logical_or.at(agg, (np.repeat(np.arange(R), N), head.ravel()), S.ravel())         # head active if it or a member sent
    cnt = np.zeros((R, N)); np.add.at(cnt, (np.repeat(np.arange(R), N), head.ravel()), S.ravel().astype(float))
    up = etx(d_sink)[None, :]
    out["leach_agg"] = float(e_member + (agg & isCH).astype(float).__mul__(up).sum())
    out["leach_fwd"] = float(e_member + (cnt * isCH * up).sum())
    out["criterion"] = float(S.sum() * E_R)
    return out

MACS = {"persistence": 0, "lightweight": 208, "cnn_lstm": 99488}
ref = sent("persistence"); Eref = radio(ref); n_ref = int(ref.sum())
res = dict(rounds=R, nodes=N, cluster_heads_per_round=nCH, reference="persistence (validation-selected on V0)",
           all_report_cost_per_report_J={k: v / (R * N) for k, v in radio(np.ones((R, N), bool)).items()},
           reference_radio_J=Eref, models={})
for m in ("lightweight", "cnn_lstm"):
    rows = []
    for s in range(20):
        S = sent(f"{m}_s{s}"); E = radio(S); extra = int(S.sum()) - n_ref
        d_inf = (MACS[m] - MACS["persistence"]) * E_OP * R * N
        rows.append({k: dict(net_saving_per_node_round_uJ=-(E[k] - Eref[k] + d_inf) / (R * N) * 1e6, radio_diff_J=E[k] - Eref[k],
                             radio_per_extra_report_J=((E[k] - Eref[k]) / extra) if extra else None) for k in E} | {"extra_reports": extra})
    res["models"][m] = dict(per_seed=rows, mean={k: dict(net_saving_per_node_round_uJ=float(np.mean([r[k]["net_saving_per_node_round_uJ"] for r in rows])),
                            radio_per_extra_report_J=float(np.mean([r[k]["radio_per_extra_report_J"] for r in rows if r[k]["radio_per_extra_report_J"] is not None])))
                            for k in ("direct", "leach_agg", "leach_fwd", "criterion")},
                            pooled_over_seeds={k: dict(radio_per_extra_report_J=float(sum(r[k]["radio_diff_J"] for r in rows) / sum(r["extra_reports"] for r in rows)))
                            for k in ("direct", "leach_agg", "leach_fwd", "criterion")},
                            seeds_with_positive_net_saving={k: [s for s, r in enumerate(rows) if r[k]["net_saving_per_node_round_uJ"] > 0] for k in ("direct", "leach_agg", "leach_fwd", "criterion")},
                            all_seeds_negative={k: all(r[k]["net_saving_per_node_round_uJ"] < 0 for r in rows) for k in ("direct", "leach_agg", "leach_fwd", "criterion")})
json.dump(res, open("radio_replay.json", "w"), indent=1)
print("all-report cost per report:", {k: f"{v:.3e}" for k, v in res["all_report_cost_per_report_J"].items()})
for m, x in res["models"].items():
    print(m, {k: (round(v["net_saving_per_node_round_uJ"], 2), f"{v['radio_per_extra_report_J']:.2e}") for k, v in x["mean"].items()}, x["all_seeds_negative"])
    print("   pooled radio per extra report:", {k: f"{v['radio_per_extra_report_J']:.2e}" for k, v in x["pooled_over_seeds"].items()})
