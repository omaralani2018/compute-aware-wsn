"""Version 1.6. Checks that every power trace in iotlab/data/ and iotlab/fixed/data/ is a contiguous block of lines of
the testbed's full consumption record of its run (iotlab_consumption_records.tar.gz, as copied from the IoT-LAB
frontends on 14 October 2026; RECORDS.sha256 lists the SHA-256 of each record). Run in caps/iotlab/records:
    python3 check_windows.py"""
import os, tarfile, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); IOT = os.path.dirname(HERE)
WIN = {"data/m3_cons_cut.txt": "454471/consumption/m3_21.oml", "data/m3_cons_cut2.txt": "454471/consumption/m3_21.oml",
       "data/m3b_cons_cut.txt": "454553/consumption/m3_22.oml", "data/nrf_cons_cut.txt": "454508/consumption/nrf52840dk_5.oml",
       "data/nrfb_cons_cut.txt": "454554/consumption/nrf52840dk_6.oml", "fixed/data/fix_m3_21.txt": "454575/consumption/m3_21.oml",
       "fixed/data/fix_m3_22.txt": "454575/consumption/m3_22.oml"}
want = dict(l.split()[::-1] for l in open(os.path.join(HERE, "RECORDS.sha256")))
rec = {}
with tarfile.open(os.path.join(HERE, "iotlab_consumption_records.tar.gz")) as t:
    for m in t.getmembers():
        if m.isfile():
            b = t.extractfile(m).read(); assert hashlib.sha256(b).hexdigest() == want[m.name], m.name
            rec[m.name] = b.decode().splitlines()
bad = 0
for win, raw in WIN.items():
    c = open(os.path.join(IOT, win)).read().splitlines(); r = rec[raw]; i = r.index(c[0]); ok = r[i:i + len(c)] == c
    bad += not ok
    print(f"{win}: lines {i + 1} to {i + len(c)} of {raw} ({len(r)} lines): {'identical' if ok else 'DIFFERENT'}")
print("records match RECORDS.sha256; windows differing:", bad)
