#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash UniCom_Decoder/scripts/run.sh --config-file /path/to/config.yaml
EOF
}

CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${CONFIG_FILE}" ]]; then
    usage
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: config file not found: ${CONFIG_FILE}"
    exit 1
fi

CONFIG_FILE=$(cd "$(dirname "${CONFIG_FILE}")" && pwd)/$(basename "${CONFIG_FILE}")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${PACKAGE_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "${PROJECT_ROOT}"

mapfile -t DEVICE_LIST < <("${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
import sys
from pathlib import Path
import yaml


def load_config(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = data.pop("base", None)
    if base is not None:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = load_config(base_path.resolve())
        merged.update(data)
        return merged
    return data


config_path = Path(sys.argv[1]).resolve()
config = load_config(config_path)
device = config.get("device")

if isinstance(device, int):
    device = [f"cuda:{device}"]
elif isinstance(device, str):
    text = device.strip()
    device = [item.strip() for item in text.split(",") if item.strip()] if "," in text else [text]
elif not isinstance(device, list):
    raise SystemExit("Config field 'device' must be an int, string, or list")

if not isinstance(device, list) or not device:
    raise SystemExit("Multi-GPU mode requires 'device' to be a non-empty list or comma-separated string")

for item in device:
    print(item)
PY
)

if [[ ${#DEVICE_LIST[@]} -eq 1 ]]; then
    "${PYTHON_BIN}" UniCom_Decoder/infer_csv.py --config "${CONFIG_FILE}"
    exit $?
fi

TMP_DIR=$(mktemp -d)
PIDS=()

cleanup() {
    local exit_code=${1:-0}
    local pid
    trap - EXIT INT TERM
    for pid in "${PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    if [[ -n "${TMP_DIR:-}" && -d "${TMP_DIR}" ]]; then
        rm -rf "${TMP_DIR}"
    fi
    exit "${exit_code}"
}

trap 'echo "Interrupted, stopping decoder workers..."; cleanup 130' INT TERM
trap 'cleanup $?' EXIT

NUM_SHARDS=${#DEVICE_LIST[@]}

for SHARD_INDEX in "${!DEVICE_LIST[@]}"; do
    DEVICE_NAME="${DEVICE_LIST[$SHARD_INDEX]}"
    WORKER_NAME=${DEVICE_NAME//:/}
    WORKER_NAME=${WORKER_NAME//,/_}
    WORKER_CONFIG="${TMP_DIR}/worker_${SHARD_INDEX}.yaml"
    cat > "${WORKER_CONFIG}" <<EOF
base: ${CONFIG_FILE}
device: ${DEVICE_NAME}
num_shards: ${NUM_SHARDS}
shard_index: ${SHARD_INDEX}
worker_name: ${WORKER_NAME}
progress_position: ${SHARD_INDEX}
EOF

    "${PYTHON_BIN}" UniCom_Decoder/infer_csv.py --config "${WORKER_CONFIG}" &
    PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAIL=1
    fi
done

exit "${FAIL}"
