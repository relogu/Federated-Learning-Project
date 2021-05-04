#!/bin/bash

python3 server.py --rounds=1000 --n_clients=2 --address=127.0.0.1:51550 & 
sleep 2 # Sleep for 2s to give the server enough time to start
python3 client.py --client_id=0 --n_samples=300 --n_clients=2 --plot=true --dump_curve=true --server=127.0.0.1:51550 &
python3 client.py --client_id=1 --n_samples=300 --n_clients=2 --plot=true --dump_curve=true --server=127.0.0.1:51550 &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400