#!/bin/bash


# unsupervosed deep embedding using EUROMDS
python3 py/server.py --address=[::]:51552 --strategy=fed_avg_k-means --kmeans_epochs=100 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --server=[::]:51552 --client_id=0 --alg=k-means --n_clients=8 --groups=1 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=1 --alg=k-means --n_clients=8 --groups=1 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=2 --alg=k-means --n_clients=8 --groups=1 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=3 --alg=k-means --n_clients=8 --groups=1 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=4 --alg=k-means --n_clients=8 --groups=1 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=5 --alg=k-means --n_clients=8 --groups=1 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=6 --alg=k-means --n_clients=8 --groups=1 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=7 --alg=k-means --n_clients=8 --groups=1 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=EUROMDS-rrrr_k-means --in_folder="$PWD/output2" 
sleep 10

# unsupervosed deep embedding using EUROMDS
python3 py/server.py --address=[::]:51552 --strategy=fed_avg_k-means --kmeans_epochs=100 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --server=[::]:51552 --client_id=0 --alg=k-means --n_clients=8 --groups=2 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=1 --alg=k-means --n_clients=8 --groups=2 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=2 --alg=k-means --n_clients=8 --groups=2 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=3 --alg=k-means --n_clients=8 --groups=2 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=4 --alg=k-means --n_clients=8 --groups=2 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=5 --alg=k-means --n_clients=8 --groups=2 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=6 --alg=k-means --n_clients=8 --groups=2 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=7 --alg=k-means --n_clients=8 --groups=2 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=EUROMDS-rrr_k-means --in_folder="$PWD/output2"
sleep 10

# unsupervosed deep embedding using EUROMDS
python3 py/server.py --address=[::]:51552 --strategy=fed_avg_k-means --kmeans_epochs=100 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --server=[::]:51552 --client_id=0 --alg=k-means --n_clients=8 --groups=3 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=1 --alg=k-means --n_clients=8 --groups=3 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=2 --alg=k-means --n_clients=8 --groups=3 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=3 --alg=k-means --n_clients=8 --groups=3 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=4 --alg=k-means --n_clients=8 --groups=3 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=5 --alg=k-means --n_clients=8 --groups=3 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=6 --alg=k-means --n_clients=8 --groups=3 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=7 --alg=k-means --n_clients=8 --groups=3 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=EUROMDS-rr_k-means --in_folder="$PWD/output2"
sleep 10

# unsupervosed deep embedding using EUROMDS
python3 py/server.py --address=[::]:51552 --strategy=fed_avg_k-means --kmeans_epochs=100 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --server=[::]:51552 --client_id=0 --alg=k-means --n_clients=8 --groups=4 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=1 --alg=k-means --n_clients=8 --groups=4 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=2 --alg=k-means --n_clients=8 --groups=4 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=3 --alg=k-means --n_clients=8 --groups=4 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=4 --alg=k-means --n_clients=8 --groups=4 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=5 --alg=k-means --n_clients=8 --groups=4 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=6 --alg=k-means --n_clients=8 --groups=4 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=7 --alg=k-means --n_clients=8 --groups=4 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=EUROMDS-r_k-means --in_folder="$PWD/output2"
sleep 10


# unsupervosed deep embedding using EUROMDS
python3 py/server.py --address=[::]:51552 --strategy=fed_avg_k-means --kmeans_epochs=100 --n_clients=8 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 py/client.py --server=[::]:51552 --client_id=0 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=1 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=2 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=3 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=4 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=5 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=6 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &
python3 py/client.py --server=[::]:51552 --client_id=7 --alg=k-means --n_clients=8 --n_clusters=10 --dataset=EUROMDS --out_fol="$PWD/output2" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
python3 scripts/plot_metrics.py --prefix=EUROMDS_k-means --in_folder="$PWD/output2"
sleep 10