#!/usr/bin/env bash
# Full pipeline in the cloud session: data -> tokenizer -> tokens -> training -> export.
# Every step is skipped when its output exists; training always resumes from
# checkpoints/latest.pt, so the script can simply be started again after an interruption.
#
#   nohup scripts/train-cloud.sh > checkpoints/pipeline.log 2>&1 &
#
# Environment: EPOCHS (default 2), TRAIN_FLAGS (extra flags for tinylaw.train, e.g. --bf16).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
EPOCHS=${EPOCHS:-2}
TRAIN_FLAGS=${TRAIN_FLAGS:---bf16}
mkdir -p checkpoints

[ -f data/raw/wikisource.jsonl ] && [ -f data/raw/Czech_legal_code.csv ] \
  || "$PY" -m tinylaw.data download
[ -f data/corpus/train.txt ] || "$PY" -m tinylaw.data build
[ -f data/tokenizer.json ] || "$PY" -m tinylaw.tokenizer train
[ -f data/tokens/train.bin ] || "$PY" -m tinylaw.tokenizer encode
# shellcheck disable=SC2086
"$PY" -m tinylaw.train --epochs "$EPOCHS" --resume $TRAIN_FLAGS
"$PY" -m tinylaw.export
echo "pipeline finished"
