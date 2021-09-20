#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir "$PWD/output_fed_udec"
# UDEC using EUROMDS
python3 clustering/py/server.py --address=[::]:51550 --n_clusters 8 --strategy=k-fed --kmeans_epochs=1 --ae_epochs=1 --cluster_epochs=10000 --n_clients=2 --out_fol="$PWD/output_fed_udec" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51550 --update_interval 100 --n_clients=2 --tied --client_id=0 --alg=udec --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec" &
python3 clustering/py/client_euromds.py --n_clusters 8 --ran_flip 0.25 --server=[::]:51550 --update_interval 100 --n_clients=2 --tied --client_id=1 --alg=udec --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --out_fol="$PWD/output_fed_udec" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/8kFED_ui2CEUROMDSfinal_deno_ude2k1u10k"
mv "$PWD/output_fed_udec/"/* "$PWD/results/8kFED_ui2CEUROMDSfinal_deno_ude2k1u10k"/
#python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSfinal_deno_ude20k1u40k --in_folder="$PWD/results/EUROMDSfinal_deno_ude20k1u40k"
sleep 10
rmdir "$PWD/output_fed_udec"

# mkdir "$PWD/output_fed_udec"
# # UDEC using EUROMDS
# python3 clustering/py/server.py --address=[::]:51550 --strategy=k-fed --kmeans_epochs=1 --ae_epochs=10000 --cluster_epochs=10000 --n_clients=4 --out_fol="$PWD/output_fed_udec" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --ran_flip 0.2 --server=[::]:51550 --update_interval 200000 --tied --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --ran_flip 0.2 --server=[::]:51550 --update_interval 200000 --tied --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --ran_flip 0.2 --server=[::]:51550 --update_interval 200000 --tied --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --ran_flip 0.2 --server=[::]:51550 --update_interval 200000 --tied --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/kFED_noui4CEUROMDSfinal_deno_ude10k1u10k"
# mv "$PWD/output_fed_udec/"/* "$PWD/results/kFED_noui4CEUROMDSfinal_deno_ude10k1u10k"/
# #python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSfinal_deno_ude20k1u40k --in_folder="$PWD/results/EUROMDSfinal_deno_ude20k1u40k"
# sleep 10
# rmdir "$PWD/output_fed_udec"

# mkdir "$PWD/output_fed_udec"
# # UDEC using EUROMDS
# python3 clustering/py/server.py --address=[::]:51550 --strategy=k-fed --kmeans_epochs=1 --ae_epochs=10000 --cluster_epochs=10000 --n_clients=6 --out_fol="$PWD/output_fed_udec" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=4 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=5 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/kFED_noui6CEUROMDSfinal_deno_ude10k1u10k"
# mv "$PWD/output_fed_udec/"/* "$PWD/results/kFED_noui6CEUROMDSfinal_deno_ude10k1u10k"/
# #python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSfinal_deno_ude20k1u40k --in_folder="$PWD/results/EUROMDSfinal_deno_ude20k1u40k"
# sleep 10
# rmdir "$PWD/output_fed_udec"

# mkdir "$PWD/output_fed_udec"
# # UDEC using EUROMDS
# python3 clustering/py/server.py --address=[::]:51550 --strategy=k-fed --kmeans_epochs=1 --ae_epochs=10000 --cluster_epochs=10000 --n_clients=6 --out_fol="$PWD/output_fed_udec" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=4 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=5 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=6 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
# python3 clustering/py/client_euromds.py --server=[::]:51550 --update_interval 200000 --tied --client_id=7 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/kFED_noui8CEUROMDSfinal_deno_ude10k1u10k"
# mv "$PWD/output_fed_udec/"/* "$PWD/results/kFED_noui8CEUROMDSfinal_deno_ude10k1u10k"/
# #python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSfinal_deno_ude20k1u40k --in_folder="$PWD/results/EUROMDSfinal_deno_ude20k1u40k"
# sleep 10
# rmdir "$PWD/output_fed_udec"
