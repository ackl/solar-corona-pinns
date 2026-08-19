#!/bin/bash
# Login-node setup for the NF2 SHARP run (needs internet; no GPU needed here).
# Validates the PINN image and downloads the SHARP FITS from JSOC.
#
#   Usage:  JSOC_EMAIL=you@inst.org ./setup.sh
#
# Re-runnable: NF2 internally handles an already downloaded time range

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
. "$PROJECT_ROOT/hpc/env.sh"
hpc_create_directories
export SOLAR_NF2_RUN_DIR="${SOLAR_NF2_RUN_DIR:-$PINN_RUN_ROOT/sharp_cea}"
export SOLAR_NF2_DATA_DIR="${SOLAR_NF2_DATA_DIR:-$SHARP_DATA_DIR}"

# JSOC download parameters (override any of these as env vars).
export JSOC_EMAIL="${JSOC_EMAIL:?set JSOC_EMAIL to your JSOC-registered email}"
export SOLAR_NF2_SHARP_NUM="${SOLAR_NF2_SHARP_NUM:-377}"
export SOLAR_NF2_T_START="${SOLAR_NF2_T_START:-2011-02-15T00:00:00}"
export SOLAR_NF2_T_END="${SOLAR_NF2_T_END:-2011-02-15T00:12:00}"
export SOLAR_NF2_SEGMENTS="${SOLAR_NF2_SEGMENTS:-[\"Br\",\"Bp\",\"Bt\",\"Br_err\",\"Bp_err\",\"Bt_err\"]}"

[[ -f "${PROJECT_IMAGE}" ]] || {
    echo "ERROR: no project image at ${PROJECT_IMAGE}" >&2
    echo "install the release images with apptainer/pull.sh" >&2
    exit 1
}

echo "=== verifying NF2 runtime ==="
"$PROJECT_ROOT/hpc/python.sh" - <<'PY'
from importlib.metadata import version

import torch

cuda = torch.version.cuda  # reliable on a GPU-less node, unlike get_arch_list()
print("NF2", version("nf2"), "| torch", torch.__version__, "| CUDA", cuda)
assert cuda and cuda.split(".")[0] == "12", f"expected CUDA 12.x, got {cuda}"
PY

echo "=== downloading SHARP $SOLAR_NF2_SHARP_NUM from JSOC -> $SOLAR_NF2_DATA_DIR ==="
"$PROJECT_ROOT/hpc/python.sh" \
    -m nlfff.pinn.run_nf2_sharp --stage download

echo "=== OK! ==="
