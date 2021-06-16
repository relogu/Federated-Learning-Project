#!/bin/bash

# simple k-means with blobs in lda distribution
python3 py/server.py --strategy=fed_avg_k-means --kmeans_epochs=15 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --client_id=0 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=1 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=2 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=3 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=4 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=5 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=6 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=7 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=lda_k-means
sleep 10

# simple k-means with blobs
python3 py/server.py --strategy=fed_avg_k-means --kmeans_epochs=15 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --client_id=0 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=1 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=2 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=3 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=4 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=5 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=6 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=7 --n_samples=70000 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=blobs &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=k-means
sleep 10

# k-FED with ae clustering
python3 py/server.py --strategy=k-fed --ae_epochs=300 --cluster_epochs=1000 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --client_id=0 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=1 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=2 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=3 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=4 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=5 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=6 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --dataset=blobs &
python3 py/client.py --client_id=7 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --dataset=blobs &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=k-fed_ae
sleep 10

# k-FED with ae clustering and lda partitions
python3 py/server.py --strategy=k-fed --ae_epochs=300 --cluster_epochs=1000 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --client_id=0 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=1 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=2 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=3 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=4 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=5 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=6 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
python3 py/client.py --client_id=7 --n_samples=70000 --alg=k_fed-ae_clust --n_clients=8 --n_clusters=10 --lda=True --dataset=blobs &
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=lda_k-fed_ae