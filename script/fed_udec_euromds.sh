#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

mkdir "$PWD/output_fed_udec"
# unsupervosed deep embedding using EUROMDS
python3 clustering/py/server.py --address=[::]:51551 --strategy=fed_avg --kmeans_epochs=1 --ae_epochs=20000 --cluster_epochs=40000 --n_clients=8 --out_fol="$PWD/output_fed_udec" & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=0 --alg=udec --binary --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=1 --alg=udec --binary --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=2 --alg=udec --binary --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=3 --alg=udec --binary --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=4 --alg=udec --binary --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=5 --alg=udec --binary --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=6 --alg=udec --binary --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &
python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=7 --alg=udec --binary --shuffle=True --fold_n=0 --n_clients=8 --groups=Genetics --groups=CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D --n_clusters=6 --out_fol="$PWD/output_fed_udec" &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait
mkdir "$PWD/results/EUROMDSrrr_ude20k1u40k"
mv "$PWD/output_fed_udec/"/* "$PWD/results/EUROMDSrrr_binary_ude20k1u40k"/
#python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSrrr_binary_ude20k1u40k --in_folder="$PWD/output1"
sleep 10
rmdir "$PWD/output_fed_udec"

# # unsupervosed deep embedding using EUROMDS
# python3 clustering/py/server.py --address=[::]:51551 --strategy=fed_avg --kmeans_epochs=1 --ae_epochs=20000 --cluster_epochs=40000 --n_clients=8 --out_fol="$PWD/output1" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=4 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=5 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=6 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=7 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=3 --n_clusters=6 --out_fol="$PWD/output1" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/EUROMDSrr_ude20k1u40k"
# mv "$PWD/output1/"/* "$PWD/results/EUROMDSrr_ude20k1u40k"/
# #python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSrr_ude20k1u40k --in_folder="$PWD/output1"
# sleep 10

# # unsupervosed deep embedding using EUROMDS
# python3 clustering/py/server.py --address=[::]:51551 --strategy=fed_avg --kmeans_epochs=1 --ae_epochs=20000 --cluster_epochs=40000 --n_clients=8 --out_fol="$PWD/output1" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=4 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=5 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=6 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=7 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=4 --n_clusters=6 --out_fol="$PWD/output1" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/EUROMDSr_ude20k1u40k"
# mv "$PWD/output1/"/* "$PWD/results/EUROMDSr_ude20k1u40k"/
# #python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSr_ude20k1u40k --in_folder="$PWD/output1"
# sleep 10


# # unsupervosed deep embedding using EUROMDS
# python3 clustering/py/server.py --address=[::]:51551 --strategy=fed_avg --kmeans_epochs=1 --ae_epochs=20000 --cluster_epochs=40000 --n_clients=8 --out_fol="$PWD/output1" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=4 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=5 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=6 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=7 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --n_clusters=6 --out_fol="$PWD/output1" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# mkdir "$PWD/results/EUROMDS_ude20k1u40k"
# mv "$PWD/output1/"/* "$PWD/results/EUROMDS_ude20k1u40k"/
# #python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDS_ude20k1u40k --in_folder="$PWD/output1"
# sleep 10

# unsupervosed deep embedding using EUROMDS
# python3 clustering/py/server.py --address=[::]:51551 --strategy=fed_avg --kmeans_epochs=1 --ae_epochs=20000 --cluster_epochs=40000 --n_clients=8 --out_fol="$PWD/output1" & 
# sleep 2 # Sleep for 2s to give the server enough time to start
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=0 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=1 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=1 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=1 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=2 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=1 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=3 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=1 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=4 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=1 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=5 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=1 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=6 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=1 --n_clusters=6 --out_fol="$PWD/output1" &
# python3 clustering/py/client_euromds.py --server=[::]:51551 --client_id=7 --alg=udec --shuffle=True --fold_n=0 --n_clients=8 --groups=1 --n_clusters=6 --out_fol="$PWD/output1" &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait
# python3 py/scripts/plot_metrics.py -f="$PWD" --prefix=EUROMDSrrrr_ude20k1u40k --in_folder="$PWD/output1" 
# sleep 10
