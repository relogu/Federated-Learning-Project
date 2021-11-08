#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Number of clusters $1"
CLUSTERS="$1"
echo "Output folder $2"
OUT_FOL="$2"

SCRIPT="py/dec_mnist.py"

mkdir "$PWD/$OUT_FOL"

nohup python3 $SCRIPT \
    --n_clusters $CLUSTERS \
    --folder "$PWD/$OUT_FOL" \
    --hardware_acc \
    --verbose >> "$PWD/$OUT_FOL/log.txt"
wait
