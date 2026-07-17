#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SOURCE_DIR="weights/common/official/PP-en_rec_ppocr_v5"
OUTPUT="weights/line_squeeze/rec_ppocrv5en_v1.onnx"

test -f "$SOURCE_DIR/inference.json"
test -f "$SOURCE_DIR/inference.pdiparams"
test ! -e "$OUTPUT" || {
  echo "目标模型已存在，按模型版本规范不覆盖: $OUTPUT" >&2
  exit 1
}
conda run -n ppocr paddle2onnx \
  --model_dir "$SOURCE_DIR" \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file "$OUTPUT" \
  --opset_version 17 \
  --enable_onnx_checker True \
  --optimize_tool onnxoptimizer
echo "已导出: $OUTPUT"
