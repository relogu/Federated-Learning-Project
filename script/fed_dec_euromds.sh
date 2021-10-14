#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Number of clusters $1"
CLUSTERS="$1"
echo "Letter defining model set up $2"
LETTER="$2"
echo "Port for server communication $3"
PORT="[::]:$3"
echo "Output folder $4"
OUT_FOL="$4"
echo "Fill NaNs $5"
FILL=""
DS=""
if [ $5 == "fill" ]
then
FILL="--fill"
DS="fill"
fi
echo "Dataset type chosen $DS"
echo "Max number of clients to simulate $6"
MAX_N_CLIENTS=$6

DATASET="--groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D"
SERVER="clustering/py/server.py"
CLIENT="clustering/py/client_euromds.py"

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

training_setup

mkdir "$PWD/$OUT_FOL"
for (( j=2; j<=$MAX_N_CLIENTS; j++ )) # stopped at 8, 9 todo
do
N_CLIENTS=$j
echo "Simulating the federated set up with $N_CLIENTS clients."

# UDEC using EUROMDS
nohup time python3 $SERVER \
    --address $PORT \
    --n_clusters $CLUSTERS \
    --strategy k-fed \
    --kmeans_epochs 1 --ae_epochs $AE_EPOCHS --cluster_epochs 10000 \
    --n_clients $N_CLIENTS \
    --out_fol "$PWD/$OUT_FOL" >> "$PWD/$OUT_FOL/server_log.txt" & 
sleep 2 # Sleep for 2s to give the server enough time to start

for (( i=0; i<$N_CLIENTS; i++ ))
do
echo "Launching client $i."
nohup time python3 $CLIENT \
    $U_NORM --tied --dropout $DROPOUT --ran_flip $RAN_FLIP \
    $DATASET $DS \
    --n_clusters $CLUSTERS \
    --server $PORT \
    --n_clients $N_CLIENTS \
    --client_id $i \
    --out_fol "$PWD/$OUT_FOL" \
    --verbose >> "$PWD/$OUT_FOL/client_${i}_log.txt" &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DEC_{$N_CLIENTS}CEUROMDS{$DS}_FED_{$LETTER}_K{$CLUSTERS}"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_{$N_CLIENTS}CEUROMDS{$DS}_FED_{$LETTER}_K{$CLUSTERS}"/
sleep 10

done
rmdir "$PWD/$OUT_FOL"
