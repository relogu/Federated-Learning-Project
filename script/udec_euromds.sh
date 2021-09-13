#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir "$PWD/output_udec"
# entire dataset
python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.02 --cl_lr 0.01 --update_interval 1000 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=1 --folder="$PWD/output_udec" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u1u_do001_rf002"
mv "$PWD/output_udec"/* "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u1u_do001_rf002"/
sleep 10
rmdir "$PWD/output_udec"

mkdir "$PWD/output_udec"
# entire dataset
python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.03 --cl_lr 0.01 --update_interval 1000 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=1 --folder="$PWD/output_udec" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u1u_do001_rf003"
mv "$PWD/output_udec"/* "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u1u_do001_rf003"/
sleep 10
rmdir "$PWD/output_udec"

mkdir "$PWD/output_udec"
# entire dataset
python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.04 --cl_lr 0.01 --update_interval 1000 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=1 --folder="$PWD/output_udec" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u1u_do001_rf004"
mv "$PWD/output_udec"/* "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u1u_do001_rf004"/
sleep 10
rmdir "$PWD/output_udec"