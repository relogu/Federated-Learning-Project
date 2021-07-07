#!/bin/bash

echo $PWD

# entire dataset
#python3 py/clustergan.py -s=euromds -n=5000 -a
#wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan
sleep 10

# entire dataset
python3 py/clustergan.py -g=4 -s=euromds -n=5000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan_r
sleep 10

# entire dataset
python3 py/clustergan.py -g=3 -s=euromds -n=5000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan_rr
sleep 10

# entire dataset
python3 py/clustergan.py -g=2 -s=euromds -n=5000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan_rrr
sleep 10

# entire dataset
python3 py/clustergan.py -g=1 -s=euromds -n=5000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_clustergan_rrrr
