cd "$(dirname "$0")"
for kind in cnnlstm_small mlp_mid; do for ds in "$@"; do python3 mid_run.py $ds $kind; done; done 2>&1 | grep --line-buffered -v "I0000\|cuda\|oneDNN\|WARNING\|provenance"
echo LANEDONE
