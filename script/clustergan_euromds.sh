#!/bin/bash

echo $PWD

# entire dataset
python3 py/clustergan.py -g=1 -s=euromds -n=10000 -a -d=18 --n_clusters=6
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_clustergan_rrrr10k
sleep 10

# entire dataset
python3 py/clustergan.py -g=2 -s=euromds -n=10000 -a -d=18 --n_clusters=6
wait
python3 clustering/scripts/plot_metrics.py  -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_clustergan_rrr10k
sleep 10

# entire dataset
python3 py/clustergan.py -g=3 -s=euromds -n=10000 -a -d=18 --n_clusters=6
wait
python3 clustering/scripts/plot_metrics.py  -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_clustergan_rr10k
sleep 10

# entire dataset
python3 py/clustergan.py -g=4 -s=euromds -n=10000 -a -d=18 --n_clusters=6
wait
python3 clustering/scripts/plot_metrics.py  -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_clustergan_r10k
sleep 10

# entire dataset
python3 py/clustergan.py -s=euromds -n=10000 -a -d=18 --n_clusters=6
wait
python3 clustering/scripts/plot_metrics.py  -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_clustergan10k
sleep 10
