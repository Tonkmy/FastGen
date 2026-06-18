#!/usr/bin/env bash
set -euo pipefail
cd /data/chenqingzhan/FastGen
LOG_DIR=FASTGEN_OUTPUT/fastgen/wan_dmd2/wan21_t2v_dmd2_OpenVid_global_8/logs/eval_10prompts
mkdir -p "$LOG_DIR"
log="$LOG_DIR/sweep_nohup_$(date +%Y%m%d_%H%M%S).log"
nohup bash fastgen/configs/experiments/WanT2V/run_infer_dmd2_eval_sweep.sh > "$log" 2>&1 &
pid=$!
echo "$pid" > "$LOG_DIR/sweep.pid"
echo "$log" > "$LOG_DIR/sweep.logpath"
echo "pid=$pid log=$log"
