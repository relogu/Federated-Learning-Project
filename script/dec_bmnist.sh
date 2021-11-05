#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Number of clusters $1"
CLUSTERS="$1"
echo "Output folder $2"
OUT_FOL="$2"
echo "Fill NaNs $3"
FILL=""
DS="final"
if [ $3 == "fill" ]
then
FILL="--fill"
DS="$3"
fi
echo "Dataset type chosen $DS"
echo "Use orthogonality constraint $4"
ORTHO=""
if [ $4 == "y" ]
then
ORTHO="--ortho"
fi

LETTER=""
SCRIPT="py/dec_bmnist.py"


mkdir "$PWD/$OUT_FOL"
LETTER="z"
DROPOUT="0.20"
RAN_FLIP="0.20"
U_NORM="--u_norm"
AE_EPOCHS="5000"
# entire dataset

nohup time python3 $SCRIPT $FILL $ORTHO \
    --ae_epochs $AE_EPOCHS \
    --cl_epochs 20000 \
    --update_interval 140 \
    --n_clusters $CLUSTERS \
    --dropout $DROPOUT \
    --ran_flip $RAN_FLIP \
    --tied $DATASET \
    --folder "$PWD/$OUT_FOL" \
    --hardware_acc \
    --verbose >> "$PWD/$OUT_FOL/log.txt"
wait
