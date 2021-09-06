#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

# clustergan with ae clustering using EUROMDS reduced twice
python3 clustering/py/server.py --address=[::]:51552 --strategy=clustergan --total_epochs=10000 --n_clients=8 --out_fol="$PWD/output2" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=0 --alg=clustergan --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output2" &
python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=1 --alg=clustergan --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output2" &
python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=2 --alg=clustergan --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output2" &
python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=3 --alg=clustergan --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output2" &
python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=4 --alg=clustergan --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output2" &
python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=5 --alg=clustergan --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output2" &
python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=6 --alg=clustergan --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output2" &
python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=7 --alg=clustergan --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output2" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSrrrr_clustergan10k --in_folder="$PWD/output2"
sleep 10

# # clustergan with ae clustering using EUROMDS reduced twice
# python3 clustering/py/server.py --address=[::]:51552 --strategy=clustergan --total_epochs=10000 --n_clients=8 --out_fol="$PWD/output2" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=0 --alg=clustergan --fold_n=0 --n_clients=8 --groups=2 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=1 --alg=clustergan --fold_n=0 --n_clients=8 --groups=2 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=2 --alg=clustergan --fold_n=0 --n_clients=8 --groups=2 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=3 --alg=clustergan --fold_n=0 --n_clients=8 --groups=2 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=4 --alg=clustergan --fold_n=0 --n_clients=8 --groups=2 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=5 --alg=clustergan --fold_n=0 --n_clients=8 --groups=2 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=6 --alg=clustergan --fold_n=0 --n_clients=8 --groups=2 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=7 --alg=clustergan --fold_n=0 --n_clients=8 --groups=2 --n_clusters=6 --out_fol="$PWD/output2" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSrrr_clustergan10k --in_folder="$PWD/output2"
# sleep 10

# # clustergan with ae clustering using EUROMDS reduced twice
# python3 clustering/py/server.py --address=[::]:51552 --strategy=clustergan --total_epochs=10000 --n_clients=8 --out_fol="$PWD/output2" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=0 --alg=clustergan --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=1 --alg=clustergan --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=2 --alg=clustergan --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=3 --alg=clustergan --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=4 --alg=clustergan --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=5 --alg=clustergan --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=6 --alg=clustergan --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=7 --alg=clustergan --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output2" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSrr_clustergan10k --in_folder="$PWD/output2"
# sleep 10

# # clustergan with ae clustering using EUROMDS reduced once
# python3 clustering/py/server.py --address=[::]:51552 --strategy=clustergan --total_epochs=10000 --n_clients=8 --out_fol="$PWD/output2" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=0 --alg=clustergan --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=1 --alg=clustergan --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=2 --alg=clustergan --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=3 --alg=clustergan --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=4 --alg=clustergan --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=5 --alg=clustergan --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=6 --alg=clustergan --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=7 --alg=clustergan --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output2" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSr_clustergan10k --in_folder="$PWD/output2"
# sleep 10

# # clustergan with ae clustering using EUROMDS
# python3 clustering/py/server.py --address=[::]:51552 --strategy=clustergan --total_epochs=10000 --n_clients=8 --out_fol="$PWD/output2" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=0 --alg=clustergan --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=1 --alg=clustergan --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=2 --alg=clustergan --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=3 --alg=clustergan --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=4 --alg=clustergan --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=5 --alg=clustergan --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=6 --alg=clustergan --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output2" &
# python3 clustering/py/client_euromds.py --server=[::]:51552 --client_id=7 --alg=clustergan --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output2" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan10k --in_folder="$PWD/output2"
# sleep 10