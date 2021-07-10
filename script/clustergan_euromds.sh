#!/bin/bash

echo $PWD

# entire dataset
python3 py/clustergan.py -g=1 -s=euromds -n=100 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan_rrrr5k
sleep 10

# entire dataset
python3 py/clustergan.py -g=2 -s=euromds -n=100 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan_rrr5k
sleep 10

# entire dataset
python3 py/clustergan.py -g=3 -s=euromds -n=100 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan_rr5k
sleep 10

# entire dataset
python3 py/clustergan.py -g=4 -s=euromds -n=100 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan_r5k
sleep 10

# entire dataset
python3 py/clustergan.py -s=euromds -n=100 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan5k
sleep 10
