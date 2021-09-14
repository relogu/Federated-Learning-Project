#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir "$PWD/output_udec1"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec1/encoder.npz"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec1/ae_history"
# entire dataset
python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.01 --update_interval 1000 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec1" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tied_ui1kEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
mv "$PWD/output_udec1"/* "$PWD/results/tied_ui1kEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
sleep 10
rmdir "$PWD/output_udec1"

mkdir "$PWD/output_udec1"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec1/encoder.npz"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec1/ae_history"
# entire dataset
python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.01 --update_interval 2000 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec1" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tied_ui2kEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
mv "$PWD/output_udec1"/* "$PWD/results/tied_ui2kEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
sleep 10
rmdir "$PWD/output_udec1"

mkdir "$PWD/output_udec1"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec1/encoder.npz"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec1/ae_history"
# entire dataset
python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.01 --update_interval 20002 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec1" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tied_nouiEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
mv "$PWD/output_udec1"/* "$PWD/results/tied_nouiEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
sleep 10
rmdir "$PWD/output_udec1"