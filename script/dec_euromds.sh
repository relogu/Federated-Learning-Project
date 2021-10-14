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

DATASET="--groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D"
LETTER=""
SCRIPT="py/udec/main.py"

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
for LETTER in "a" "b" "c" "d" "e" "f" "g"
do
training_setup
# entire dataset
nohup time python3 $SCRIPT $FILL $ORTHO \
    --ae_epochs $AE_EPOCHS \
    --cl_epochs 20000 \
    --n_clusters $CLUSTERS \
    --dropout $DROPOUT \
    --ran_flip $RAN_FLIP \
    --tied $DATASET \
    --folder "$PWD/$OUT_FOL" \
    --hardware_acc \
    --verbose >> "$PWD/$OUT_FOL/log.txt"
wait
sleep 5
mkdir "$PWD/results/DEC_EUROMDS{$DS}_single_{$LETTER}_{$ORTHO}_K$CLUSTERS"
mv "$PWD/$OUT_FOL"/* "$PWD/results/DEC_EUROMDS{$DS}_single_{$LETTER}_{$ORTHO}_K$CLUSTERS"/
sleep 10
done
rmdir "$PWD/$OUT_FOL"
