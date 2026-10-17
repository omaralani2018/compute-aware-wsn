"""PREREG17 host check: decisions of the firmware's own code (ARM build under qemu-arm) against the Python emulation
(gen.py float64, gen2.py bit-exact integer). Writes expected_decisions.json, used by the analysis to check the node's D lines."""
import json, zlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); NX = os.path.dirname(HERE)
H = json.load(open(os.path.join(NX, "netx_host.json"))); PL = json.load(open(os.path.join(NX, "netx_plan.json"))); R = H["R"]
def unbits(h): b = bytes.fromhex(h); return [(b[r >> 3] >> (r & 7)) & 1 for r in range(R)]
exp = {}; bad = 0
for b in ("m3_float", "m3_int", "nrf"):
    L = open(os.path.join(HERE, f"out_{b}.txt")).read().split("\n")
    refs = list(map(int, L[0].split()[2:])); exp[b] = {}
    for l in L[1:]:
        if not l.startswith("D "): continue
        _, key, ntx, hx = l.split(); ds, arm = key.split(":"); d = unbits(hx); assert sum(d) == int(ntx)
        if b == "m3_int":
            if arm not in PL["streams"][ds]["arms_int"]: continue
            want = unbits(PL["streams"][ds]["arms_int"][arm]["bits"])
            ndiff = sum(x != y for x, y in zip(d, want)); ok = ndiff == 0
        else:
            ok = "%08x" % zlib.crc32(bytes(d)) == H["streams"][ds]["arms"][arm]["crc32"]; ndiff = None if ok else "crc"
        if arm in (H["streams"][ds]["ref"][b], "lightweight", "cnn_lstm"): exp[b][key] = {"tx": sum(d), "bits": hx}
        print("%-8s %-28s tx %3d  %s" % (b, key, sum(d), "MATCH" if ok else "DIFF %s" % ndiff)); bad += not ok
    ARM = {0: "persistence", 1: "expsm", 2: "linear_extrap"}
    for i, ds in enumerate(["taxi", "exchange_cpc", "machine_temp", "cpu"]):
        assert ARM[refs[i]] == H["streams"][ds]["ref"][b], (b, ds)
json.dump(exp, open(os.path.join(HERE, "expected_decisions.json"), "w"), indent=1)
print("mismatches:", bad)
