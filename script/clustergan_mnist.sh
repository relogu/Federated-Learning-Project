#!/bin/bash

echo $PWD

# entire dataset
python3 py/clustergan.py -n=1000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=MNIST_clustergan1k
sleep 10

# entire dataset
python3 py/clustergan.py -n=2000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=MNIST_clustergan2k
sleep 10

# entire dataset
python3 py/clustergan.py -n=3000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=MNIST_clustergan3k
sleep 10

# entire dataset
python3 py/clustergan.py -n=4000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=MNIST_clustergan4k
sleep 10

# entire dataset
python3 py/clustergan.py -n=5000 -a
wait
python3 clustering/scripts/plot_metrics.py -f="$PWD" --prefix=MNIST_clustergan5k
