#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

# entire dataset
python3 py/udec/main.py --groups=1 --n_clusters=6 --ae_epochs=20 --cl_epochs=10 --folder="$PWD/output"
wait
python3 py/scripts/plot_metrics.py -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrrrr_single_ude20k1u10k
sleep 10

# entire dataset
python3 py/udec/main.py --groups=2 --n_clusters=6 --ae_epochs=20000 --cl_epochs=10000 --folder="$PWD/output"
wait
python3 py/scripts/plot_metrics.py  -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrrr_single_ude20k1u10k
sleep 10

# entire dataset
python3 py/udec/main.py --groups=3 --n_clusters=6 --ae_epochs=20000 --cl_epochs=10000 --folder="$PWD/output"
wait
python3 py/scripts/plot_metrics.py  -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSrr_single_ude20k1u10k
sleep 10

# entire dataset
python3 py/udec/main.py --groups=4 --n_clusters=6 --ae_epochs=20000 --cl_epochs=10000 --folder="$PWD/output"
wait
python3 py/scripts/plot_metrics.py  -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDSr_single_ude20k1u10k
sleep 10

# entire dataset
python3 py/udec/main.py --n_clusters=6 --ae_epochs=20000 --cl_epochs=10000 --folder="$PWD/output"
wait
python3 py/scripts/plot_metrics.py  -f="$PWD" --out_folder="$PWD" --in_folder="$PWD/output" --prefix=EUROMDS_single_ude20k1u10k
sleep 10
