#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"
CLUSTERS="8"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# entire dataset
python3 py/udec/main.py --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.20 --ran_flip 0.20 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_a_K$CLUSTERS"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_a_K$CLUSTERS"/
sleep 10
rmdir "$PWD/output_udec2"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# entire dataset
python3 py/udec/main.py --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.05 --ran_flip 0.05 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_b_K$CLUSTERS"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_b_K$CLUSTERS"/
sleep 10
rmdir "$PWD/output_udec2"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.05 --ran_flip 0.05 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_c_K$CLUSTERS"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_c_K$CLUSTERS"/
sleep 10
rmdir "$PWD/output_udec2"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.20 --ran_flip 0.20 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_d_K$CLUSTERS"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_d_K$CLUSTERS"/
sleep 10
rmdir "$PWD/output_udec2"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.10 --ran_flip 0.10 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_e_K$CLUSTERS"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_e_K$CLUSTERS"/
sleep 10
rmdir "$PWD/output_udec2"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.05 --ran_flip 0.05 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_f_K$CLUSTERS"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_f_K$CLUSTERS"/
sleep 10
rmdir "$PWD/output_udec2"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.01 --ran_flip 0.01 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_g_K$CLUSTERS"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_g_K$CLUSTERS"/
sleep 10
rmdir "$PWD/output_udec2"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# entire dataset
python3 py/udec/main.py --u_norm --ae_epochs 5000 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.01 --ran_flip 0.01 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_h_K$CLUSTERS"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_h_K$CLUSTERS"/
sleep 10
rmdir "$PWD/output_udec2"
