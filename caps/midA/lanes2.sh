cd "$(dirname "$0")"
for ds in "$@"; do python3 mid_run.py $ds mlp_mid; done 2>&1 | grep --line-buffered -v "I0000\|cuda\|oneDNN\|WARNING\|provenance"
echo LANEDONE
