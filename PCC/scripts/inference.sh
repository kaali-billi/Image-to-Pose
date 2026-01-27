#!/usr/bin/env bash

set -x
GPUS=$1
PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=${GPUS} python tools/inference.py --main ${PY_ARGS}