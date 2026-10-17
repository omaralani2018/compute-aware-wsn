cd "$(dirname "$0")"
while [ $(cat laneA.log laneB.log 2>/dev/null | grep -c LANEDONE) != 2 ]; do sleep 60; done
python3 analyse_lb24.py > analyse_lb24.log 2>&1; echo ANALYSED >> analyse_lb24.log
