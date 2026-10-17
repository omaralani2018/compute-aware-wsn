S=$(cd "$(dirname "$0")/.." && pwd)
while [ $(cat $S/confB/laneA.log $S/confB/laneB.log 2>/dev/null | grep -c LANEDONE) != 2 ]; do sleep 60; done
cd $S/lb24
for ds in "$@"; do python3 lb24_run.py $ds cheap; python3 lb24_run.py $ds lightweight; done 2>&1 | grep --line-buffered -v "I0000\|cuda\|oneDNN\|WARNING\|provenance"
echo LANEDONE
