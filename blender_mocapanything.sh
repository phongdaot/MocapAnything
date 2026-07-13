#!/usr/bin/env bash
set -euo pipefail

# 发布版:通过环境变量配置,默认用 PATH 上的 blender。
#   export BLENDER_BIN=/path/to/blender        # 你的 blender 可执行文件(建议 4.x/5.x)
#   export MOCAP_ENV_LIB=/path/to/conda/env/lib # 可选:补 conda 环境的 lib 到 LD_LIBRARY_PATH
BLENDER_BIN="${BLENDER_BIN:-blender}"
MOCAP_ENV_LIB="${MOCAP_ENV_LIB:-}"

if [ -n "${MOCAP_ENV_LIB}" ]; then
  export LD_LIBRARY_PATH="${MOCAP_ENV_LIB}:${LD_LIBRARY_PATH:-}"
fi
exec "${BLENDER_BIN}" "$@"
