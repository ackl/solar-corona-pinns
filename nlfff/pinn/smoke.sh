#!/bin/bash
#   Prereq:  ./setup.sh has downloaded the FITS data

set -euo pipefail

SUBMIT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SUBMIT_DIR/../.." && pwd)"
. "$PROJECT_ROOT/hpc/env.sh"
. "$PROJECT_ROOT/hpc/slurm.sh"
hpc_create_directories
SMOKE_DIR="${PINN_SMOKE_RUN_DIR:-$PINN_RUN_ROOT/smoke}"

[ -d "$SHARP_DATA_DIR" ] || { echo "no FITS at $SHARP_DATA_DIR -- run $SUBMIT_DIR/setup.sh first"; exit 1; }

mkdir -p "$SMOKE_DIR"

echo "==== pls work ===="
export SOLAR_NF2_RUN_DIR="$SMOKE_DIR"
export SOLAR_NF2_DATA_DIR="$SHARP_DATA_DIR"
export SOLAR_NF2_WANDB_MODE=offline
export SOLAR_NF2_MODEL_DIM=64 SOLAR_NF2_EPOCHS=1 SOLAR_NF2_ITERATIONS=200
export SOLAR_NF2_VALIDATION_PIXEL_PER_DS=16
hpc_srun gpu --time=00:15:00 bash --noprofile --norc -c '
  set -euo pipefail
  echo "host: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  "'"$PROJECT_ROOT"'/hpc/python.sh" \
      -m nlfff.pinn.run_nf2_sharp --stage train
  "'"$PROJECT_ROOT"'/hpc/python.sh" \
      -m nlfff.pinn.run_nf2_sharp --stage export
'
echo "==== OK! ===="
