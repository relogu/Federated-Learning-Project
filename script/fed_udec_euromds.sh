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
AE_EPOCHS="5000"
fi

mkdir "$PWD/$OUT_FOL"
# UDEC using EUROMDS
python3 clustering/py/server.py --address=$PORT --n_clusters $CLUSTERS --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=2 --out_fol="$PWD/$OUT_FOL" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 clustering/py/client_euromds.py $U_NORM --verbose --cuda --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=2 --tied --client_id=0 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --cuda --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=2 --tied --client_id=1 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DECFED_2CEUROMDSfinal_deno_fed_$LETTER-K$CLUSTERS"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DECFED_2CEUROMDSfinal_deno_fed_$LETTER-K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# UDEC using EUROMDS
python3 clustering/py/server.py --address=$PORT --n_clusters $CLUSTERS --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=4 --out_fol="$PWD/$OUT_FOL" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=4 --tied --client_id=0 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=4 --tied --client_id=1 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=4 --tied --client_id=2 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=4 --tied --client_id=3 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DECFED_4CEUROMDSfinal_deno_fed_$LETTER-K$CLUSTERS"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DECFED_4CEUROMDSfinal_deno_fed_$LETTER-K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# UDEC using EUROMDS
python3 clustering/py/server.py --n_clusters $CLUSTERS --address=$PORT --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=6 --out_fol="$PWD/$OUT_FOL" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=6 --tied --client_id=0 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=6 --tied --client_id=1 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=6 --tied --client_id=2 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=6 --tied --client_id=3 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=6 --tied --client_id=4 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=6 --tied --client_id=5 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DECFED_6CEUROMDSfinal_deno_fed_$LETTER-K$CLUSTERS"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DECFED_6CEUROMDSfinal_deno_fed_$LETTER-K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# UDEC using EUROMDS
python3 clustering/py/server.py --n_clusters $CLUSTERS --address=$PORT --strategy=k-fed --kmeans_epochs=1 --ae_epochs=$AE_EPOCHS --cluster_epochs=10000 --n_clients=8 --out_fol="$PWD/$OUT_FOL" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=8 --tied --client_id=0 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=8 --tied --client_id=1 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=8 --tied --client_id=2 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=8 --tied --client_id=3 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=8 --tied --client_id=4 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=8 --tied --client_id=5 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=8 --tied --client_id=6 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &
python3 clustering/py/client_euromds.py $U_NORM --verbose --n_clusters $CLUSTERS --dropout $DROPOUT --ran_flip $RAN_FLIP --server=$PORT --update_interval 100 --n_clients=8 --tied --client_id=7 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/$OUT_FOL" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/DECFED_8CEUROMDSfinal_deno_fed_$LETTER-k$CLUSTERS"
mv "$PWD/$OUT_FOL/"/* "$PWD/results/DECFED_8CEUROMDSfinal_deno_fed_$LETTER-k$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"
