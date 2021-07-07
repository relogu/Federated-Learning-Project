#!/bin/bash

# clustergan and lda partitions
python3 py/server.py --strategy=clustergan --total_epochs=2000 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --client_id=0 --alg=clustergan --n_clients=8 --n_clusters=10 --dataset=EUROMDS &
python3 py/client.py --client_id=1 --alg=clustergan --n_clients=8 --n_clusters=10 --dataset=EUROMDS &
python3 py/client.py --client_id=2 --alg=clustergan --n_clients=8 --n_clusters=10 --dataset=EUROMDS &
python3 py/client.py --client_id=3 --alg=clustergan --n_clients=8 --n_clusters=10 --dataset=EUROMDS &
python3 py/client.py --client_id=4 --alg=clustergan --n_clients=8 --n_clusters=10 --dataset=EUROMDS &
python3 py/client.py --client_id=5 --alg=clustergan --n_clients=8 --n_clusters=10 --dataset=EUROMDS &
python3 py/client.py --client_id=6 --alg=clustergan --n_clients=8 --n_clusters=10 --dataset=EUROMDS &
python3 py/client.py --client_id=7 --alg=clustergan --n_clients=8 --n_clusters=10 --dataset=EUROMDS &
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=EUROMDS-redredred_clustergan
