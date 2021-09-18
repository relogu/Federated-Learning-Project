#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"


# mkdir "$PWD/output_fed_udec3"
# # UDEC using EUROMDS
# python3 clustering/py/server.py --address=[::]:51553 --strategy=k-fed --kmeans_epochs=1 --ae_epochs=10000 --cluster_epochs=10000 --n_clients=6 --out_fol="$PWD/output_fed_udec3" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51553 --update_interval 500 --tied --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec3" &
# python3 clustering/py/client_euromds.py --server=[::]:51553 --update_interval 500 --tied --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec3" &
# python3 clustering/py/client_euromds.py --server=[::]:51553 --update_interval 500 --tied --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec3" &
# python3 clustering/py/client_euromds.py --server=[::]:51553 --update_interval 500 --tied --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec3" &
# python3 clustering/py/client_euromds.py --server=[::]:51553 --update_interval 500 --tied --client_id=4 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec3" &
# python3 clustering/py/client_euromds.py --server=[::]:51553 --update_interval 500 --tied --client_id=5 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec3" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/kFED_ui6CEUROMDSfinal_deno_ude10k1u10k"
# mv "$PWD/output_fed_udec3/"/* "$PWD/results/kFED_ui6CEUROMDSfinal_deno_ude10k1u10k"/
# #python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSfinal_deno_ude20k1u40k --in_folder="$PWD/results/EUROMDSfinal_deno_ude20k1u40k"
# sleep 10
# rmdir "$PWD/output_fed_udec3"

mkdir "$PWD/output_fed_udec3"
# UDEC using EUROMDS
python3 clustering/py/server.py --address=[::]:51553 --strategy=k-fed --kmeans_epochs=1 --ae_epochs=10000 --cluster_epochs=1 --n_clients=8 --out_fol="$PWD/output_fed_udec3" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51553 --update_interval 500 --tied --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec3" &
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51553 --update_interval 500 --tied --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec3" &
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51553 --update_interval 500 --tied --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec3" &
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51553 --update_interval 500 --tied --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec3" &
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51553 --update_interval 500 --tied --client_id=4 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec3" &
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51553 --update_interval 500 --tied --client_id=5 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec3" &
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51553 --update_interval 500 --tied --client_id=6 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec3" &
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51553 --update_interval 500 --tied --client_id=7 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec3" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/kFED_ui8CEUROMDSfinal_deno_ude10k1u10k"
mv "$PWD/output_fed_udec3/"/* "$PWD/results/kFED_ui8CEUROMDSfinal_deno_ude10k1u10k"/
#python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSfinal_deno_ude20k1u40k --in_folder="$PWD/results/EUROMDSfinal_deno_ude20k1u40k"
sleep 10
rmdir "$PWD/output_fed_udec3"
