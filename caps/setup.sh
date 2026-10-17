#!/bin/bash
# Version 1.6. Rebuilds, inside caps/, what the code in caps/ was run against, from the rest of this package:
#   caps/stage17/run  the main run directory (as the README setup), with the NAB files under nab_repo/data
#   caps/stb_exec     the Grand St Bernard segments, summary and per-run results (Section 6.10)
#   caps/confB/run_<deployment>/stb_segments_obs.pkl   the four SensorScope preparations (Amendment 7a)
#   the links nab_repo (adaptA/, tolA/, cltA/, lb24/) and intel_time.npy (adaptA/, lb24/)
# The Candanedo files of Sections 6.6 and 6.13 are not redistributed: put them in caps/confC/data/ (see README.md).
set -e
C=$(cd "$(dirname "$0")" && pwd); R=$(dirname "$C")
mkdir -p "$C/stage17/run" "$C/stb_exec"
cd "$C/stage17/run"
cp "$R"/src/*.py "$R"/results/*.json "$R"/inputs/*.npy "$R"/inputs/*.json "$R"/inputs/*.csv "$R"/inputs/mote_locs.txt .
cp -r "$R/weights" .
python3 -c "import nab_data,os,shutil; [(os.makedirs(os.path.dirname(p),exist_ok=True),shutil.copy('$R/inputs/nab/'+os.path.basename(p),p)) for p in nab_data.NAB.values()]"
cd "$C/stb_exec"
cp "$R/src/segeval.py" "$R/stb/run_outputs/stb_summary.json" .
cp -r "$R/stb/run_outputs/runs" .
python3 "$R/stb/amendment_1/stb_prepare.py" "$R/inputs/sensorscope/stbernard-meteo.zip" > prep.log
for d in fishnet genepi pdg2008 plaine_morte; do
  if [ $d = plaine_morte ]; then NF=18; CV=10; else NF=17; CV=9; fi
  (cd "$C/confB/run_$d" && DEP=$d NF=$NF CT=8 CV=$CV python3 ../prep_deploy.py ../data/$d-meteo.zip > prep_setup.log)
done
(cd "$C/confB" && sha256sum -c --quiet SEGMENTS.sha256)   # the four preparations, byte for byte as the runs used them
for d in adaptA tolA cltA lb24 boot; do ln -sfn ../stage17/run/nab_repo "$C/$d/nab_repo"; done
for d in adaptA lb24; do ln -sfn ../stage17/run/intel_time.npy "$C/$d/intel_time.npy"; done
echo "setup done"
