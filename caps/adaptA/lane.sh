for ds in "$@"; do s0=0; [ "$ds" = taxi ] && s0=1; python3 retrain_lw.py $ds $s0 20 2>&1 | grep -v "^I0000\|cuda\|oneDNN\|WARNING\|provenance"; done
