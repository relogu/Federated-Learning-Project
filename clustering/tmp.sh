#!/bin/bash

# k-FED with ae clustering using EUROMDS
python3 py/server.py --strategy=k-fed --ae_epochs=1000 --cluster_epochs=1000 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --client_id=0 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=6 --dataset=EUROMDS &
python3 py/client.py --client_id=1 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=6 --dataset=EUROMDS &
python3 py/client.py --client_id=2 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=6 --dataset=EUROMDS &
python3 py/client.py --client_id=3 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=6 --dataset=EUROMDS &
python3 py/client.py --client_id=4 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=6 --dataset=EUROMDS &
python3 py/client.py --client_id=5 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=6 --dataset=EUROMDS &
python3 py/client.py --client_id=6 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=6 --dataset=EUROMDS &
python3 py/client.py --client_id=7 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=6 --dataset=EUROMDS &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=EUROMDS_k-fed_ae
sleep 10
