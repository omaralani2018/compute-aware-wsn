cd "$(dirname "$0")"
while [ $(cat laneA2.log laneB2.log 2>/dev/null | grep -c LANEDONE) != 2 ]; do sleep 60; done
python3 analyse_mid.py > analyse_mid.log 2>&1; echo ANALYSED >> analyse_mid.log
