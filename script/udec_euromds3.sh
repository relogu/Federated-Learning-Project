#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir "$PWD/output_udec3"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec3/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec3/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters 4 --dropout 0.01 --ran_flip 0.01 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec3" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/4paperDEC_EUROMDSfinal_single_g"
mv "$PWD/output_udec3"/* "$PWD/results/4paperDEC_EUROMDSfinal_single_g"/
sleep 10
rmdir "$PWD/output_udec3"

mkdir "$PWD/output_udec3"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec3/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec3/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters 7 --dropout 0.01 --ran_flip 0.01 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec3" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/7paperDEC_EUROMDSfinal_single_g"
mv "$PWD/output_udec3"/* "$PWD/results/7paperDEC_EUROMDSfinal_single_g"/
sleep 10
rmdir "$PWD/output_udec3"

mkdir "$PWD/output_udec3"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec3/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec3/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters 11 --dropout 0.01 --ran_flip 0.01 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec3" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/11paperDEC_EUROMDSfinal_single_g"
mv "$PWD/output_udec3"/* "$PWD/results/11paperDEC_EUROMDSfinal_single_g"/
sleep 10
rmdir "$PWD/output_udec3"