"""Version 1.6. Runs a script of this folder with the first emulator, qemu_v1_si20.py (written under PREREG13.txt, with a
fixed 20-bit scale for the reciprocal of the input spread), under the module name qemu, which is how gen_q.py wrote app_q/models_q.h and ref_q.json for the node firmware, and how accuracy_q.py wrote the
file now named accuracy_q_v1_fixed_si20.json (qemu.py, the later version, gives the Intel model an input scale of
22 bits instead of the firmware's 20). Run in caps/iotlab/fixed:
    python3 run_with_v1_emulator.py gen_q.py
    python3 run_with_v1_emulator.py accuracy_q.py      # writes accuracy_q.json: rename it, after saving the shipped one"""
import sys, os, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qemu_v1_si20; sys.modules["qemu"] = qemu_v1_si20
script = os.path.join(HERE, sys.argv[1]); sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
