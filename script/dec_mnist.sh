#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Arguments passed: $#"
echo "Arguments: $@"
if [ $1 == "-h" ]; then
echo "Arguments pattern: <out folder> <tied> <unit norm> <ortho weights> <uncoll feat> <use bias> <gpus ids>"
exit
else
echo "Output folder $1"
OUT_FOL="$1"
shift
echo "Tied? $1"
TIED=""
if [ $1 == "y" ]; then
TIED="--tied"
fi
shift
echo "Unit norm? $1"
UNORM=""
if [ $1 == "y" ]; then
UNORM="--u_norm"
fi
shift
echo "Ortho weights? $1"
ORTHO=""
if [ $1 == "y" ]; then
ORTHO="--ortho"
fi
shift
echo "Uncollerated features? $1"
UNCOLL=""
if [ $1 == "y" ]; then
UNCOLL="--uncoll"
fi
shift
echo "Use bias? $1"
BIAS=""
if [ $1 == "y" ]; then
BIAS="--use_bias"
fi
shift
GPUS=""
for var in "$@"
do
    echo "Selected GPU $var"
    GPUS="$GPUS $var"
done
echo "GPU selected $GPUS"

SCRIPT="py/dec_mnist.py"

mkdir "$PWD/$OUT_FOL"

nohup python3 $SCRIPT $TIED $UNORM $ORTHO $UNCOLL $BIAS\
    --folder "$PWD/$OUT_FOL" \
    --hardware_acc \
    --gpus$GPUS \
    --verbose >> "$PWD/$OUT_FOL/log.txt"
wait
fi
