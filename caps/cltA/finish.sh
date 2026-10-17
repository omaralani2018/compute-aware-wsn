while [ $(cat laneA.log laneB.log | grep -c LANEDONE) != 2 ]; do sleep 30; done
python3 analyse_clt.py > analyse_clt.log 2>&1; echo ANALYSED >> analyse_clt.log
