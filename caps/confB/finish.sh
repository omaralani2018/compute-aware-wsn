C=$(cd "$(dirname "$0")" && pwd)
while [ $(cat $C/laneA.log $C/laneB.log | grep -c LANEDONE) != 2 ]; do sleep 30; done
cd $C && OMP_NUM_THREADS=1 python3 -I analyse_confB.py > analyse_confB.log 2>&1; echo ANALYSED >> analyse_confB.log
