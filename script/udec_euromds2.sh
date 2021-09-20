#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir "$PWD/output_udec2"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec/encoder.npz"
#cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec/ae_history"
# entire dataset
python3 py/udec/main.py --n_clusters 8 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D  --folder "$PWD/output_udec2" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/paperDEC_EUROMDSfinal_single_d"
mv "$PWD/output_udec2"/* "$PWD/results/paperDEC_EUROMDSfinal_single_d"/
sleep 10
rmdir "$PWD/output_udec2"

# mkdir "$PWD/output_udec2"
# cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
# cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# # entire dataset
# python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.1 --update_interval 100 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec2" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/tied_ui100_clEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
# mv "$PWD/output_udec2"/* "$PWD/results/tied_ui100_clEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
# sleep 10
# rmdir "$PWD/output_udec2"

# mkdir "$PWD/output_udec2"
# cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/encoder.npz" "$PWD/output_udec2/encoder.npz"
# cp "$PWD/results/tied_uiEUROMDSfinal_deno_single_ude10k1u1u_do001_rf005/ae_history" "$PWD/output_udec2/ae_history"
# # entire dataset
# python3 py/udec/main.py --tied --dropout 0.01 --ran_flip 0.05 --cl_lr 0.1 --update_interval 500 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec2" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/tied_ui500_clEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"
# mv "$PWD/output_udec2"/* "$PWD/results/tied_ui500_clEUROMDSfinal_deno_single_ude10k1u20k_do001_rf005"/
# sleep 10
# rmdir "$PWD/output_udec2"
