# waits for the NAB confirmatory analysis, then runs P_obs for the four deployments on two cores
C=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$C/code PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
while ! grep -q ANALYSED $C/../confA/analyse_conf.log 2>/dev/null; do sleep 30; done
seeds() { for s in $(seq $3 $(($4 - 1))); do python3 $C/code/stb_run.py $1 $2 $s $((s + 1)) || python3 $C/code/stb_run.py $1 $2 $s $((s + 1)); done; }
if [ "$1" = A ]; then
  for d in genepi fishnet pdg2008 plaine_morte; do (cd $C/run_$d && python3 $C/code/stb_run.py obs cheap); done
  (cd $C/run_genepi && seeds obs lightweight 0 20)
  for d in fishnet pdg2008 plaine_morte; do (cd $C/run_$d && seeds obs lightweight 0 20 && seeds obs cnn_lstm 0 20); done
else
  (cd $C/run_genepi && while [ ! -f stb_obs_cheap.json ]; do sleep 20; done; seeds obs cnn_lstm 0 20)
fi
echo LANEDONE
