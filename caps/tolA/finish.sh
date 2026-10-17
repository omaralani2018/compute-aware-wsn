while [ $(cat laneA.log laneB.log | grep -c LANEDONE) != 2 ]; do sleep 30; done
python3 analyse_ta.py > analyse_ta.log 2>&1; echo ANALYSED >> analyse_ta.log
