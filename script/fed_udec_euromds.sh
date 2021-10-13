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

if [ $2 == "a" ]
then
DROPOUT="0.20"
RAN_FLIP="0.20"
U_NORM=""
AE_EPOCHS="2500"
fi

if [ $2 == "b" ]
then
DROPOUT="0.05"
RAN_FLIP="0.05"
U_NORM=""
AE_EPOCHS="2500"
fi

if [ $2 == "c" ]
then
DROPOUT="0.20"
RAN_FLIP="0.20"
U_NORM="--u_norm"
AE_EPOCHS="2500"
fi

if [ $2 == "d" ]
then
DROPOUT="0.10"
RAN_FLIP="0.10"
U_NORM="--u_norm"
AE_EPOCHS="2500"
fi

if [ $2 == "e" ]
then
DROPOUT="0.05"
RAN_FLIP="0.05"
U_NORM="--u_norm"
AE_EPOCHS="2500"
fi

if [ $2 == "f" ]
then
DROPOUT="0.01"
RAN_FLIP="0.01"
U_NORM="--u_norm"
AE_EPOCHS="2500"
fi

if [ $2 == "g" ]
then
DROPOUT="0.01"
RAN_FLIP="0.01"
U_NORM="--u_norm"
AE_EPOCHS="1"
fi

mkdir "$PWD/$OUT_FOL"
for (( j=2; j<=$MAX_N_CLIENTS; j++ ))
do
N_CLIENTS=$j
echo "Simulating the federated set up with $N_CLIENTS clients."

# UDEC using EUROMDS
python3 $SERVER --address $PORT --n_clusters $CLUSTERS --strategy k-fed --kmeans_epochs 1 --ae_epochs $AE_EPOCHS --cluster_epochs 10000 --n_clients $N_CLIENTS --out_fol "$PWD/$OUT_FOL" & 
sleep 2 # Sleep for 2s to give the server enough time to start

for (( i=0; i<$N_CLIENTS; i++ ))
do
echo "Launching client $i."
python3 $CLIENT $DS $U_NORM --tied --dropout $DROPOUT --ran_flip $RAN_FLIP --n_clusters $CLUSTERS --server $PORT --n_clients $N_CLIENTS --client_id $i $DATASET --out_fol "$PWD/$OUT_FOL" --verbose &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DEC_{$N_CLIENTS}CEUROMDS{$DS}_FED_{$LETTER}_K{$CLUSTERS}"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_{$N_CLIENTS}CEUROMDS{$DS}_FED_{$LETTER}_K{$CLUSTERS}"/
sleep 10

done
rmdir "$PWD/$OUT_FOL"
