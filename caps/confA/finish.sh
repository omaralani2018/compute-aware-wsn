while ! grep -q LANEDONE laneB.log; do sleep 30; done
bash lane.sh TravelTime_387:cnn_lstm > laneC.log 2>&1
while [ $(cat laneA.log laneC.log | grep -c LANEDONE) != 2 ]; do sleep 30; done
python3 -I analyse_conf.py > analyse_conf.log 2>&1; echo ANALYSED >> analyse_conf.log
