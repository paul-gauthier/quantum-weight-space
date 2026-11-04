#!/usr/bin/env bash
# GPU pipeline for one lattice size: fine-tune ours, check it from mixed starts, then one guided Lanczos step in shards.
# usage: bash scripts/ft_pipeline.sh L LR STEPS SEED CHUNK SHARDS [SHARD_SEED0]
#   fp64 MinSR, 2,048 samples, as in the 14x14/18x18 runs; the Lanczos shards share one offset c (the run's final
#   10-step mean energy) so they can be pooled with scripts/pool_lanczos.py. Logs to runs/ft and runs/lanczos.
#   The shards sample the guide of scripts/ft_lanczos.py (--guide-floor 0.02), as for every reported Lanczos energy.
set -uo pipefail
cd "$(dirname "$0")/.."
l=$1; lr=$2; steps=$3; seed=$4; chunk=$5; shards=$6; s0=${7:-$((seed * 10))}
name=ft${l}_ours_lr${lr}
PY="env -u LD_LIBRARY_PATH PYTHONPATH=. .venv-gpu/bin/python -u"
mkdir -p runs/ft runs/lanczos
$PY scripts/finetune.py ours $l --steps $steps --lr $lr --diag-shift 1e-4 --samples 2048 --therm 20 --eval-chains 2048 \
  --eval-sweeps 3 --chunk $chunk --chunk-bwd $chunk --eval-chunk 8192 --fp64 --seed $seed --output runs/ft/$name.json \
  > runs/ft/$name.log 2>&1 || exit 1
$PY scripts/ft_eval.py runs/ft/$name.json --chains 8192 --sweeps 40 --every 10 --chunk 8192 > runs/ft/${name}_mixed.log 2>&1
c=$(python3 -c "import json; t=json.load(open('runs/ft/$name.json'))['trace'][-10:]; print(round(sum(x['E'] for x in t)/len(t)*4*$l*$l, 3))")
echo "c = $c"
for k in $(seq 0 $((shards - 1))); do
  $PY scripts/ft_lanczos.py runs/ft/$name.json --chains 2048 --sweeps 40 --guide-floor 0.02 --seed $((s0 + k)) --c $c \
    --output runs/lanczos/${name}_s$((s0 + k)).npz > runs/lanczos/${name}_s$((s0 + k)).log 2>&1
done
echo PIPELINE_DONE
