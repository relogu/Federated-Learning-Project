#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

# OVERFIT
# mkdir "$PWD/output_udec"
# # entire dataset
# python3 py/udec/main.py --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=20000 --folder="$PWD/output_udec" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/tied_orthoEUROMDSfinal_deno_single_ude10k1u20k"
# mv "$PWD/output_udec"/* "$PWD/results/EUROMDSfinal_deno_single_ude10k1u20k"/
# sleep 10
# rmdir "$PWD/output_udec"

mkdir "$PWD/output_udec"
# entire dataset
python3 py/udec/main.py --dropout --cl_lr 0.01 --update_interval 1000 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=10 --ae_epochs=10000 --cl_epochs=10000 --folder="$PWD/output_udec" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/tied_uiEUROMDSfinal10_deno_single_ude10k1u10k"
mv "$PWD/output_udec"/* "$PWD/results/tied_uiEUROMDSfinal10_deno_single_ude10k1u10k"/
sleep 10
rmdir "$PWD/output_udec"

# mkdir "$PWD/output_udec"
# # entire dataset
# python3 py/udec/main.py --cl_lr 0.01 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=40000 --folder="$PWD/output_udec" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/EUROMDSfinal_deno_single_ude10k1u40k"
# mv "$PWD/output_udec"/* "$PWD/results/EUROMDSfinal_deno_single_ude10k1u40k"/
# sleep 10
# rmdir "$PWD/output_udec"

# mkdir "$PWD/output_udec"
# # entire dataset
# python3 py/udec/main.py --cl_lr 0.01 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=40000 --folder="$PWD/output_udec" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u40k"
# mv "$PWD/output_udec"/* "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u40k"/
# sleep 10
# rmdir "$PWD/output_udec"

# mkdir "$PWD/output_udec"
# # entire dataset
# python3 py/udec/main.py --cl_lr 0.001 --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=60000 --folder="$PWD/output_udec" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/EUROMDSfinal_deno_single_ude10k1u60k"
# mv "$PWD/output_udec"/* "$PWD/results/EUROMDSfinal_deno_single_ude10k1u60k"/
# sleep 10
# rmdir "$PWD/output_udec"

# mkdir "$PWD/output_udec"
# # entire dataset
# python3 py/udec/main.py --cl_lr 0.001 --tied --groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --ae_epochs=10000 --cl_epochs=60000 --folder="$PWD/output_udec" --hardware_acc
# wait
# sleep 5
# mkdir "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u60k"
# mv "$PWD/output_udec"/* "$PWD/results/tiedEUROMDSfinal_deno_single_ude10k1u60k"/
# sleep 10
# rmdir "$PWD/output_udec"

# # entire dataset
# python3 py/udec/main.py --groups=2 --n_clusters=6 --ae_epochs=20000 --cl_epochs=10000 --folder="$PWD/output"
# wait
# python3 py/scripts/plot_metrics.py  -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrrr_single_ude20k1u10k
# sleep 10

# # entire dataset
# python3 py/udec/main.py --groups=3 --n_clusters=6 --ae_epochs=20000 --cl_epochs=10000 --folder="$PWD/output"
# wait
# python3 py/scripts/plot_metrics.py  -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrr_single_ude20k1u10k
# sleep 10

# # entire dataset
# python3 py/udec/main.py --groups=4 --n_clusters=6 --ae_epochs=20000 --cl_epochs=10000 --folder="$PWD/output"
# wait
# python3 py/scripts/plot_metrics.py  -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSr_single_ude20k1u10k
# sleep 10

# # entire dataset
# python3 py/udec/main.py --n_clusters=6 --ae_epochs=20000 --cl_epochs=10000 --folder="$PWD/output"
# wait
# python3 py/scripts/plot_metrics.py  -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_single_ude20k1u10k
# sleep 10