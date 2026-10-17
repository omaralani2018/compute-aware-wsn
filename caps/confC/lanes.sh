S=$(cd "$(dirname "$0")/.." && pwd)
while [ $(cat $S/confB/laneA.log $S/confB/laneB.log 2>/dev/null | grep -c LANEDONE) != 2 ]; do sleep 60; done
cd $S/confC
for job in "$@"; do ds=${job%%:*}; k=${job##*:}; python3 conf_train.py $ds $k; done 2>&1 | grep --line-buffered -v "I0000\|cuda\|oneDNN\|WARNING\|provenance"
echo LANEDONE
