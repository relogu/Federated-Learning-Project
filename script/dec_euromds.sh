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
echo "Use loss $5"
echo "AE epochs $6"

DATASET="--groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D"
LETTER=""
SCRIPT="py/dec_euromds.py"

training_setup () {
    echo "The setup selected for training is $LETTER"
    # defining autoencoder epochs
    if [ $LETTER == "a" ] || [ $LETTER == "b" ] || [ $LETTER == "c" ] || [ $LETTER == "d" ] || [ $LETTER == "e" ] || [ $LETTER == "f" ]
    then
    AE_EPOCHS="2500"
    else
    AE_EPOCHS="5000"
    fi

    # defining unit norm constraint
    if [ $LETTER == "c" ] || [ $LETTER == "d" ] || [ $LETTER == "e" ] || [ $LETTER == "f" ] || [ $LETTER == "g" ]
    then
    U_NORM="--u_norm"
    else
    U_NORM=""
    fi

    # defining d/o and r/f frequencies
    if [ $LETTER == "a" ] || [ $LETTER == "c" ]
    then
    DROPOUT="0.20"
    RAN_FLIP="0.20"
    elif [ $LETTER == "d" ]
    then
    DROPOUT="0.10"
    RAN_FLIP="0.10"
    elif [ $LETTER == "b" ] || [ $LETTER == "e" ]
    then
    DROPOUT="0.05"
    RAN_FLIP="0.05"
    elif [ $LETTER == "f" ] || [ $LETTER == "g" ]
    then
    DROPOUT="0.01"
    RAN_FLIP="0.01"
    fi
}



mkdir "$PWD/$OUT_FOL"
# for LETTER in "a" "b" "c" "d" "e" "f" "g"
# do
# training_setup
LETTER="z"
DROPOUT="0.20"
RAN_FLIP="0.20"
U_NORM="--u_norm"
AE_EPOCHS=$6

nohup time python3 $SCRIPT $FILL $ORTHO \
    --ae_epochs $AE_EPOCHS \
    --ae_loss $5 \
    --cl_epochs 20000 \
    --update_interval 3 \
    --n_clusters $CLUSTERS \
    --dropout $DROPOUT \
    --ran_flip $RAN_FLIP \
    --tied $DATASET \
    --folder "$PWD/$OUT_FOL" \
    --hardware_acc \
    --verbose >> "$PWD/$OUT_FOL/log.txt"
wait
sleep 5
