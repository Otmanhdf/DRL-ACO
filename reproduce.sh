#!/usr/bin/env bash
# Full experimental pipeline of the paper, from scratch.
# Run from the repository root:  bash reproduce.sh
# Total time is roughly 8-10 hours on 2 CPU cores (training and the 1,000-task
# evaluations dominate). Every step is deterministic (fixed seeds), so the
# CSV files in results/ are reproduced exactly on the same library versions.
# To regenerate only the statistics, tables and figures from the shipped
# results, run the last block ("4. analysis") alone.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results policies tables figures data
PY=${PY:-python3}

# 0. dataset (instances are generated from seeds; this writes them to data/)
$PY src/export_dataset.py

# 1. fixed-parameter tuning on validation instances (Table "tuning", Fig. 7)
for n in 100 300 500 700 1000; do $PY src/tune.py $n > logs/tune_$n.log 2>&1; done

# 2. training: main policy (5 seeds) and ablation policies
for s in 0 1 2 3 4; do $PY src/train.py drlaco $s > logs/train_drlaco_$s.log 2>&1; done
for f in a b r ab ar br; do $PY src/train.py free_$f 0 --free $f > logs/train_free_$f.log 2>&1; done
$PY src/train.py act27  0 --actions A27  > logs/train_act27.log 2>&1
$PY src/train.py act252 0 --actions A252 > logs/train_act252.log 2>&1
$PY src/train.py negF   0 --reward negF  > logs/train_negF.log 2>&1
$PY src/train.py K5     0 --K 5          > logs/train_K5.log 2>&1
$PY src/train.py K20    0 --K 20         > logs/train_K20.log 2>&1
$PY src/train.py small  0 --scales 100,300 > logs/train_small.log 2>&1

# 3. evaluation (30 test instances per setting)
for n in 100 300 500 700 1000; do $PY src/evaluate.py main $n > logs/main_$n.log 2>&1; done
for n in 300 1000; do $PY src/evaluate.py main $n --workload heavy > logs/heavy_$n.log 2>&1; done
for n in 100 1000; do $PY src/evaluate.py ablation $n > logs/ablation_$n.log 2>&1; done
$PY src/evaluate.py weights 300 --reps 15 > logs/weights_300.log 2>&1
$PY src/front.py 300 10 > logs/front_300.log 2>&1
$PY src/worked_example.py > logs/worked_example.log 2>&1

# 4. analysis: statistics, tables and figures
$PY src/analyze.py
$PY src/tables.py
$PY src/make_figures.py
$PY src/fig1_flow.py
echo "done"
