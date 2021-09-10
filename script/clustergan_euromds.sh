#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir "$PWD/output_clustergan"
python3 py/clustergan/main.py -g Genetics -g CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output_clustergan" --binary
wait
mkdir "$PWD/results/EUROMDSfinal_bin_single_clustergan10k"
mv "$PWD/output_clustergan"/* "$PWD/results/EUROMDSfinal_bin_single_clustergan10k"/
sleep 10
rmdir "$PWD/output_clustergan"

mkdir "$PWD/output_clustergan"
python3 py/clustergan/main.py -w -g Genetics -g CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output_clustergan" --binary
wait
mkdir "$PWD/results/EUROMDSfinal_w_bin_single_clustergan10k"
mv "$PWD/output_clustergan"/* "$PWD/results/EUROMDSfinal_w_bin_single_clustergan10k"/
sleep 10
rmdir "$PWD/output_clustergan"

mkdir "$PWD/output_clustergan"
python3 py/clustergan/main.py -g Genetics -g CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output_clustergan"
wait
mkdir "$PWD/results/EUROMDSfinal_single_clustergan10k"
mv "$PWD/output_clustergan"/* "$PWD/results/EUROMDSfinal_single_clustergan10k"/
sleep 10
rmdir "$PWD/output_clustergan"

mkdir "$PWD/output_clustergan"
python3 py/clustergan/main.py -w -g Genetics -g CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output_clustergan" --binary
wait
mkdir "$PWD/results/EUROMDSfinal_w_single_clustergan10k"
mv "$PWD/output_clustergan"/* "$PWD/results/EUROMDSfinal_w_single_clustergan10k"/
sleep 10
rmdir "$PWD/output_clustergan"

# # entire dataset
# python3 py/clustergan/main.py -g=2 -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
# wait
# python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrrr_single_clustergan10k
# sleep 10

# # entire dataset
# python3 py/clustergan/main.py -g=3 -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
# wait
# python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrr_single_clustergan10k
# sleep 10

# # entire dataset
# python3 py/clustergan/main.py -g=4 -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
# wait
# python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSr_single_clustergan10k
# sleep 10

# # entire dataset
# python3 py/clustergan/main.py -s=euromds -e=10000 -a -d=18 --n_clusters=6 --folder="$PWD/output" --binary
# wait
# python3 clustering/scripts/plot_metrics.py -f="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_single_clustergan10k
# sleep 10
