#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_textgen_smoke.sh \
  --model-root MODEL_ROOT \
  --dataset-root DATASET_ROOT \
  --model MODEL_ID \
  --output-dir OUTPUT_DIR \
  [--save SAVE_NAME] \
  [--method gumbel] \
  [--number-of-experiments 1] \
  [--batch-size 1] \
  [--tokens-count 1] \
  [--prompt-tokens 50] \
  [--buffer-tokens 20] \
  [--watermark-key-length 1000] \
  [--seed 1] \
  [--device cpu]

Run a single minimal text-generation smoke test without Slurm.
This mirrors one real command from 2-textgen-helper.sh but with a single task.

Required:
  --model-root   Root directory containing model folders such as:
                 <model-root>/facebook/opt-1.3b/model and tokenizer
  --dataset-root Root directory containing the C4 dataset train split
  --model        Hugging Face model id used by the original workflow
  --output-dir   Directory where result files will be created

Notes:
  - The script creates output_dir automatically.
  - It passes the requested path to 2-textgen.py as a safe, quoted path.
  - CPU is the default; pass --device cuda explicitly on a CUDA server.
  - Local loading errors stop the run; no automatic online dataset fallback.
  - Existing result prefixes are rejected rather than overwritten.
EOF
}

MODEL_ROOT=""
DATASET_ROOT=""
MODEL_ID=""
OUTPUT_DIR=""
SAVE_NAME="smoke"
METHOD="gumbel"
NUMBER_OF_EXPERIMENTS="1"
BATCH_SIZE="1"
TOKENS_COUNT="1"
PROMPT_TOKENS="50"
BUFFER_TOKENS="20"
WATERMARK_KEY_LENGTH="1000"
SEED="1"
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-root)
      [[ $# -ge 2 ]] || { echo "Error: --model-root requires a value." >&2; usage >&2; exit 2; }
      MODEL_ROOT="$2"; shift 2 ;;
    --dataset-root)
      [[ $# -ge 2 ]] || { echo "Error: --dataset-root requires a value." >&2; usage >&2; exit 2; }
      DATASET_ROOT="$2"; shift 2 ;;
    --model)
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value." >&2; usage >&2; exit 2; }
      MODEL_ID="$2"; shift 2 ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "Error: --output-dir requires a value." >&2; usage >&2; exit 2; }
      OUTPUT_DIR="$2"; shift 2 ;;
    --save)
      [[ $# -ge 2 ]] || { echo "Error: --save requires a value." >&2; usage >&2; exit 2; }
      SAVE_NAME="$2"; shift 2 ;;
    --method)
      [[ $# -ge 2 ]] || { echo "Error: --method requires a value." >&2; usage >&2; exit 2; }
      METHOD="$2"; shift 2 ;;
    --number-of-experiments)
      [[ $# -ge 2 ]] || { echo "Error: --number-of-experiments requires a value." >&2; usage >&2; exit 2; }
      NUMBER_OF_EXPERIMENTS="$2"; shift 2 ;;
    --batch-size)
      [[ $# -ge 2 ]] || { echo "Error: --batch-size requires a value." >&2; usage >&2; exit 2; }
      BATCH_SIZE="$2"; shift 2 ;;
    --tokens-count)
      [[ $# -ge 2 ]] || { echo "Error: --tokens-count requires a value." >&2; usage >&2; exit 2; }
      TOKENS_COUNT="$2"; shift 2 ;;
    --prompt-tokens)
      [[ $# -ge 2 ]] || { echo "Error: --prompt-tokens requires a value." >&2; usage >&2; exit 2; }
      PROMPT_TOKENS="$2"; shift 2 ;;
    --buffer-tokens)
      [[ $# -ge 2 ]] || { echo "Error: --buffer-tokens requires a value." >&2; usage >&2; exit 2; }
      BUFFER_TOKENS="$2"; shift 2 ;;
    --watermark-key-length)
      [[ $# -ge 2 ]] || { echo "Error: --watermark-key-length requires a value." >&2; usage >&2; exit 2; }
      WATERMARK_KEY_LENGTH="$2"; shift 2 ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "Error: --seed requires a value." >&2; usage >&2; exit 2; }
      SEED="$2"; shift 2 ;;
    --device)
      [[ $# -ge 2 ]] || { echo "Error: --device requires a value." >&2; exit 2; }
      DEVICE="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL_ROOT" ]]; then
  echo "Error: --model-root is required." >&2
  usage >&2
  exit 2
fi
if [[ -z "$DATASET_ROOT" ]]; then
  echo "Error: --dataset-root is required." >&2
  usage >&2
  exit 2
fi
if [[ -z "$MODEL_ID" ]]; then
  echo "Error: --model is required." >&2
  usage >&2
  exit 2
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: --output-dir is required." >&2
  usage >&2
  exit 2
fi

mkdir -p -- "$OUTPUT_DIR"
SAVE_PATH="$OUTPUT_DIR/$SAVE_NAME"

# Minimal valid values derived from the code constraints in 2-textgen.py:
# - number_of_experiments >= 1
# - batch_size >= 1
# - tokens_count >= 1 (the loop in 2-textgen.py treats it as a positive length)
# - prompt_tokens + tokens_count must be satisfiable by a dataset example; the script
#   skips examples that are too short.
if ! [[ "$NUMBER_OF_EXPERIMENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --number-of-experiments must be a positive integer >= 1." >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --batch-size must be a positive integer >= 1." >&2
  exit 2
fi
if ! [[ "$TOKENS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --tokens-count must be a positive integer >= 1." >&2
  exit 2
fi
if ! [[ "$PROMPT_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "Error: --prompt-tokens must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$BUFFER_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "Error: --buffer-tokens must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$WATERMARK_KEY_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --watermark-key-length must be a positive integer." >&2
  exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Error: --seed must be a non-negative integer." >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$PROJECT_ROOT/2-textgen.py" \
  --device "$DEVICE" \
  --save "$SAVE_PATH" \
  --model_root "$MODEL_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --model "$MODEL_ID" \
  --method "$METHOD" \
  --number_of_experiments "$NUMBER_OF_EXPERIMENTS" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --tokens_count "$TOKENS_COUNT" \
  --prompt_tokens "$PROMPT_TOKENS" \
  --buffer_tokens "$BUFFER_TOKENS" \
  --watermark_key_length "$WATERMARK_KEY_LENGTH" \
  --output_dir "$OUTPUT_DIR"
