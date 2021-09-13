#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir "$PWD/output_udec"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec/encoder.npz"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec/ae_history"
# entire dataset
python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.01 --update_interval 100 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tied_ui100EUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
mv "$PWD/output_udec"/* "$PWD/results/tied_ui100EUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
sleep 10
rmdir "$PWD/output_udec"

mkdir "$PWD/output_udec"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec/encoder.npz"
cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec/ae_history"
# entire dataset
python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.01 --update_interval 500 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tied_ui500EUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
mv "$PWD/output_udec"/* "$PWD/results/tied_ui500EUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
sleep 10
rmdir "$PWD/output_udec"

# mkdir "$PWD/output_udec"
# cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec/encoder.npz"
# cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec/ae_history"
# # entire dataset
# python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.01 --update_interval 1000 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/tied_ui1kEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
# mv "$PWD/output_udec"/* "$PWD/results/tied_ui1kEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
# sleep 10
# rmdir "$PWD/output_udec"

# mkdir "$PWD/output_udec"
# cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec/encoder.npz"
# cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec/ae_history"
# # entire dataset
# python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.01 --update_interval 2000 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/tied_ui2kEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
# mv "$PWD/output_udec"/* "$PWD/results/tied_ui2kEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
# sleep 10
# rmdir "$PWD/output_udec"