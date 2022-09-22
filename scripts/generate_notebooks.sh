#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
EXPERIMENTS_DIR="$ROOT_DIR/src/experiments/"
NOTEBOOKS_DIR="$ROOT_DIR/notebooks/"

mkdir -p "$NOTEBOOKS_DIR"

for file in "$EXPERIMENTS_DIR"*.py; do
  if [[ "$file" != *"__init__"* ]]; then
    echo "Generating jupyter notebook from '$file' module"
    poetry run sphx_glr_python_to_jupyter.py "$file";
    echo "Moving generated notebook file to notebooks directory"
    mv -- "${file%.py}.ipynb" "$NOTEBOOKS_DIR"
  fi
done
