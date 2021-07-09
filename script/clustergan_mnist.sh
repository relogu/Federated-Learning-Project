#!/bin/bash

echo $PWD
'''
# entire dataset
python3 py/clustergan.py -f="$PWD/output1" -n=1000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output1"  --prefix=MNIST_clustergan1k
sleep 10
'''
# entire dataset
python3 py/clustergan.py -f="$PWD/output1" -n=2000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output1"  --prefix=MNIST_clustergan2k
sleep 10
'''
# entire dataset
python3 py/clustergan.py -f="$PWD/output1" -n=3000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output1"  --prefix=MNIST_clustergan3k
'''