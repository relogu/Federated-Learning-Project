#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

# entire dataset
python3 py/clustergan/main.py -g=1 -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrrrr_single_clustergan10k
sleep 10

# entire dataset
python3 py/clustergan/main.py -g=2 -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrrr_single_clustergan10k
sleep 10

# entire dataset
python3 py/clustergan/main.py -g=3 -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrr_single_clustergan10k
sleep 10

# entire dataset
python3 py/clustergan/main.py -g=4 -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSr_single_clustergan10k
sleep 10

# entire dataset
python3 py/clustergan/main.py -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_single_clustergan10k
sleep 10
