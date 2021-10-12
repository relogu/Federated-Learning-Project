#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Number of clusters $1"
CLUSTERS="$1"
echo "Letter defining network set up $2"
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

# mkdir "$PWD/$OUT_FOL"
# # UDEC using EUROMDS
# python3 $SERVER --address=$PORT --n_clusters $CLUSTERS --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=1 --n_clients=2 --out_fol="$PWD/$OUT_FOL" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=2 --tied --client_id=0 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=2 --tied --client_id=1 $DATASET --out_fol="$PWD/$OUT_FOL" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/DEC_2CEUROMDS{$DS}_FED_$LETTER-K$CLUSTERS"
# mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_2CEUROMDS{$DS}_FED_$LETTER-K$CLUSTERS"/
# sleep 10
# rmdir "$PWD/$OUT_FOL"

# mkdir "$PWD/$OUT_FOL"
# # UDEC using EUROMDS
# python3 $SERVER --address=$PORT --n_clusters $CLUSTERS --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=4 --out_fol="$PWD/$OUT_FOL" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=4 --tied --client_id=0 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=4 --tied --client_id=1 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=4 --tied --client_id=2 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=4 --tied --client_id=3 $DATASET --out_fol="$PWD/$OUT_FOL" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/DEC_4CEUROMDS{$DS}_FED_$LETTER-K$CLUSTERS"
# mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_4CEUROMDS{$DS}_FED_$LETTER-K$CLUSTERS"/
# sleep 10
# rmdir "$PWD/$OUT_FOL"

# mkdir "$PWD/$OUT_FOL"
# # UDEC using EUROMDS
# python3 $SERVER --n_clusters $CLUSTERS --address=$PORT --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=6 --out_fol="$PWD/$OUT_FOL" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=6 --tied --client_id=0 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=6 --tied --client_id=1 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=6 --tied --client_id=2 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=6 --tied --client_id=3 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=6 --tied --client_id=4 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=6 --tied --client_id=5 $DATASET --out_fol="$PWD/$OUT_FOL" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/DEC_6CEUROMDS{$DS}_FED_$LETTER-K$CLUSTERS"
# mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_6CEUROMDS{$DS}_FED_$LETTER-K$CLUSTERS"/
# sleep 10
# rmdir "$PWD/$OUT_FOL"

# mkdir "$PWD/$OUT_FOL"
# # UDEC using EUROMDS
# python3 $SERVER --n_clusters $CLUSTERS --address=$PORT --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=8 --out_fol="$PWD/$OUT_FOL" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=8 --tied --client_id=0 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=8 --tied --client_id=1 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=8 --tied --client_id=2 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=8 --tied --client_id=3 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=8 --tied --client_id=4 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=8 --tied --client_id=5 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=8 --tied --client_id=6 $DATASET --out_fol="$PWD/$OUT_FOL" &
# python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=8 --tied --client_id=7 $DATASET --out_fol="$PWD/$OUT_FOL" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/DEC_8CEUROMDS{$DS}_FED_$LETTER-k$CLUSTERS"
# mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_8CEUROMDS{$DS}_FED_$LETTER-k$CLUSTERS"/
# sleep 10
# rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# UDEC using EUROMDS
python3 $SERVER --n_clusters $CLUSTERS --address=$PORT --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=10 --out_fol="$PWD/$OUT_FOL" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=0 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=1 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=2 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=3 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=4 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=5 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=6 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=7 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=8 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=10 --tied --client_id=9 $DATASET --out_fol="$PWD/$OUT_FOL" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DEC_10CEUROMDS{$DS}_FED_$LETTER-k$CLUSTERS"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_10CEUROMDS{$DS}_FED_$LETTER-k$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# UDEC using EUROMDS
python3 $SERVER --n_clusters $CLUSTERS --address=$PORT --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=12 --out_fol="$PWD/$OUT_FOL" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=0 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=1 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=2 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=3 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=4 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=5 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=6 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=7 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=8 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=9 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=10 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=12 --tied --client_id=11 $DATASET --out_fol="$PWD/$OUT_FOL" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DEC_12CEUROMDS{$DS}_FED_$LETTER-k$CLUSTERS"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_12CEUROMDS{$DS}_FED_$LETTER-k$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# UDEC using EUROMDS
python3 $SERVER --n_clusters $CLUSTERS --address=$PORT --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=14 --out_fol="$PWD/$OUT_FOL" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=0 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=1 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=2 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=3 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=4 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=5 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=6 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=7 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=8 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=9 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=10 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=11 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=12 $DATASET --out_fol="$PWD/$OUT_FOL" &
python3 $CLIENT $DS $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --n_clients=14 --tied --client_id=13 $DATASET --out_fol="$PWD/$OUT_FOL" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DEC_12CEUROMDS{$DS}_FED_$LETTER-k$CLUSTERS"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DEC_12CEUROMDS{$DS}_FED_$LETTER-k$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"
