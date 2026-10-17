"""Version 1.6. Rebuilds replication.txt: whether the second node of each type reproduces every decision of the first
(lightweight and CNN-LSTM median net saving sign and screen, per stream) and the ratio of their incremental energies.
Run in caps/iotlab:  python3 replication.py > replication.txt"""
import json
for a, b in (("m3", "m3b"), ("nrf", "nrfb")):
    A, B = (json.load(open(f"iotlab_{x}_summary.json")) for x in (a, b))
    dec = lambda J: {ds: {k: (v[k]["median_net_uJ"] > 0, v[k]["screened"]) for k in ("lightweight", "cnn_lstm")} for ds, v in J["streams"].items()}
    r = {k: round(B["per_forecast"][k]["E_inc_J"] / A["per_forecast"][k]["E_inc_J"], 4) for k in ("lightweight", "cnn_lstm")}
    print(f"{a} vs {b} decisions reproduced: {dec(A) == dec(B)} | E_inc ratio second/first: {r}")
