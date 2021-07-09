#!/bin/bash

echo $PWD

# entire dataset
python3 py/clustergan.py -f="$PWD/output2" -n=4000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output2"  --prefix=MNIST_clustergan4k
sleep 10

# entire dataset
python3 py/clustergan.py -f="$PWD/output2" -n=5000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output2"  --prefix=MNIST_clustergan5k
