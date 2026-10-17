for ds in "$@"; do python3 clt_train.py $ds 0 20 2>&1 | grep --line-buffered -v "I0000\|cuda\|oneDNN\|WARNING\|provenance"; done
echo LANEDONE
