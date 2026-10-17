for job in "$@"; do ds=${job%%:*}; k=${job##*:}; python3 conf_train.py $ds $k 2>&1 | grep --line-buffered -v "I0000\|cuda\|oneDNN\|WARNING\|provenance"; done
echo LANEDONE
