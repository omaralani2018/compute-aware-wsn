C=$(cd "$(dirname "$0")" && pwd)
while [ $(cat $C/laneA.log $C/laneB.log | grep -c LANEDONE) != 2 ]; do sleep 30; done
cd $C && python3 -I analyse_conf.py > analyse_conf.log 2>&1; echo ANALYSED >> analyse_conf.log
