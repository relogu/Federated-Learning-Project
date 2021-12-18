#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"

# python3 examples/dec_mnist.py --cuda True --gpu-id 0 --out-folder out_torch_mnist_mse &
# python3 examples/dec_mnist.py --cuda True --gpu-id 0 --out-folder out_torch_mnist_tied_mse --is-tied True --pretrain-epochs 500 --finetune-epochs 300 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 1 --out-folder out_torch_mnist_mse_sobel --ae-mod-loss sobel &
# python3 examples/dec_mnist.py --cuda True --gpu-id 1 --out-folder out_torch_mnist_tied_mse_sobel --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss sobel &
# python3 examples/dec_mnist.py --cuda True --gpu-id 2 --out-folder out_torch_mnist_mse_gausk1 --ae-mod-loss gausk1 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 2 --out-folder out_torch_mnist_tied_mse_gausk1 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk1 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 3 --out-folder out_torch_mnist_mse_gausk3 --ae-mod-loss gausk3 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 3 --out-folder out_torch_mnist_tied_mse_gausk3 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk3 &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait

# python3 examples/dec_mnist.py --cuda True --gpu-id 0 --out-folder out_torch_mnist_mse &
# python3 examples/dec_mnist.py --cuda True --gpu-id 0 --out-folder out_torch_mnist_tied_mse --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --alpha 9 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 1 --out-folder out_torch_mnist_mse_sobel --ae-mod-loss sobel --alpha 9 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 1 --out-folder out_torch_mnist_tied_mse_sobel --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss sobel --alpha 9 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 2 --out-folder out_torch_mnist_mse_gausk1 --ae-mod-loss gausk1 --alpha 9 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 2 --out-folder out_torch_mnist_tied_mse_gausk1 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk1 --alpha 9 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 3 --out-folder out_torch_mnist_mse_gausk3 --ae-mod-loss gausk3 --alpha 9 &
# python3 examples/dec_mnist.py --cuda True --gpu-id 3 --out-folder out_torch_mnist_tied_mse_gausk3 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk3 &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait

# python3 examples/dec_bmnist.py --cuda True --gpu-id 0 --out-folder out_torch_bmnist_mse &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 0 --out-folder out_torch_bmnist_tied_mse --is-tied True --pretrain-epochs 500 --finetune-epochs 300 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 1 --out-folder out_torch_bmnist_mse_sobel --ae-mod-loss sobel &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 1 --out-folder out_torch_bmnist_tied_mse_sobel --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss sobel &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 2 --out-folder out_torch_bmnist_mse_gausk1 --ae-mod-loss gausk1 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 2 --out-folder out_torch_bmnist_tied_mse_gausk1 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk1 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 3 --out-folder out_torch_bmnist_mse_gausk3 --ae-mod-loss gausk3 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 3 --out-folder out_torch_bmnist_tied_mse_gausk3 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk3 &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait

# python3 examples/dec_bmnist.py --cuda True --gpu-id 0 --out-folder out_torch_bmnist_mse &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 0 --out-folder out_torch_bmnist_tied_mse --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --alpha 9 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 1 --out-folder out_torch_bmnist_mse_sobel --ae-mod-loss sobel --alpha 9 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 1 --out-folder out_torch_bmnist_tied_mse_sobel --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss sobel --alpha 9 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 2 --out-folder out_torch_bmnist_mse_gausk1 --ae-mod-loss gausk1 --alpha 9 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 2 --out-folder out_torch_bmnist_tied_mse_gausk1 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk1 --alpha 9 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 3 --out-folder out_torch_bmnist_mse_gausk3 --ae-mod-loss gausk3 --alpha 9 &
# python3 examples/dec_bmnist.py --cuda True --gpu-id 3 --out-folder out_torch_bmnist_tied_mse_gausk3 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk3 &

# # This will allow you to use CTRL+C to stop all background processes
# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
# wait

python3 examples/dec_mnist.py --cuda True --gpu-id 0 --out-folder out_torch_mnist_bce-wl --ae-main-loss bce-wl &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 1 --out-folder out_torch_mnist_tied_bce-wl --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-main-loss bce-wl &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 2 --out-folder out_torch_mnist_bce-wl_sobel --ae-main-loss bce-wl --ae-mod-loss sobel &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 3 --out-folder out_torch_mnist_tied_bce-wl_sobel --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss sobel --ae-main-loss bce-wl &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 4 --out-folder out_torch_mnist_bce-wl_gausk1 --ae-main-loss bce-wl --ae-mod-loss gausk1 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 5 --out-folder out_torch_mnist_tied_bce-wl_gausk1 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk1 --ae-main-loss bce-wl &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 6 --out-folder out_torch_mnist_bce-wl_gausk3 --ae-main-loss bce-wl --ae-mod-loss gausk3 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 7 --out-folder out_torch_mnist_tied_bce-wl_gausk3 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk3 --ae-main-loss bce-wl &
sleep 10 &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait

python3 examples/dec_mnist.py --cuda True --gpu-id 0 --out-folder out_torch_mnist_bce-wl --ae-main-loss bce-wl --alpha 9 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 1 --out-folder out_torch_mnist_tied_bce-wl --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-main-loss bce-wl --alpha 9 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 2 --out-folder out_torch_mnist_bce-wl_sobel --ae-main-loss bce-wl --ae-mod-loss sobel --alpha 9 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 3 --out-folder out_torch_mnist_tied_bce-wl_sobel --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss sobel --ae-main-loss bce-wl --alpha 9 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 4 --out-folder out_torch_mnist_bce-wl_gausk1 --ae-main-loss bce-wl --ae-mod-loss gausk1 --alpha 9 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 5 --out-folder out_torch_mnist_tied_bce-wl_gausk1 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk1 --ae-main-loss bce-wl --alpha 9 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 6 --out-folder out_torch_mnist_bce-wl_gausk3 --ae-main-loss bce-wl --ae-mod-loss gausk3 --alpha 9 &
sleep 10 &
python3 examples/dec_mnist.py --cuda True --gpu-id 7 --out-folder out_torch_mnist_tied_bce-wl_gausk3 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk3 --ae-main-loss bce-wl --alpha 9 &
sleep 10 &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait

python3 examples/dec_bmnist.py --cuda True --gpu-id 0 --out-folder out_torch_bmnist_bce-wl --ae-main-loss bce-wl &
python3 examples/dec_bmnist.py --cuda True --gpu-id 1 --out-folder out_torch_bmnist_tied_bce-wl --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-main-loss bce-wl &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 2 --out-folder out_torch_bmnist_bce-wl_sobel --ae-main-loss bce-wl --ae-mod-loss sobel &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 3 --out-folder out_torch_bmnist_tied_bce-wl_sobel --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss sobel --ae-main-loss bce-wl &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 4 --out-folder out_torch_bmnist_bce-wl_gausk1 --ae-main-loss bce-wl --ae-mod-loss gausk1 &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 5 --out-folder out_torch_bmnist_tied_bce-wl_gausk1 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk1 --ae-main-loss bce-wl &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 6 --out-folder out_torch_bmnist_bce-wl_gausk3 --ae-main-loss bce-wl --ae-mod-loss gausk3 &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 7 --out-folder out_torch_bmnist_tied_bce-wl_gausk3 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk3 --ae-main-loss bce-wl &
sleep 10 &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait

python3 examples/dec_bmnist.py --cuda True --gpu-id 0 --out-folder out_torch_bmnist_bce-wl --ae-main-loss bce-wl --alpha 9 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 1 --out-folder out_torch_bmnist_tied_bce-wl --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-main-loss bce-wl --alpha 9 &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 2 --out-folder out_torch_bmnist_bce-wl_sobel --ae-main-loss bce-wl --ae-mod-loss sobel --alpha 9 &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 3 --out-folder out_torch_bmnist_tied_bce-wl_sobel --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss sobel --ae-main-loss bce-wl --alpha 9 &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 4 --out-folder out_torch_bmnist_bce-wl_gausk1 --ae-main-loss bce-wl --ae-mod-loss gausk1 --alpha 9 &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 5 --out-folder out_torch_bmnist_tied_bce-wl_gausk1 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk1 --ae-main-loss bce-wl --alpha 9 &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 6 --out-folder out_torch_bmnist_bce-wl_gausk3 --ae-main-loss bce-wl --ae-mod-loss gausk3 --alpha 9 &
sleep 10 &
python3 examples/dec_bmnist.py --cuda True --gpu-id 7 --out-folder out_torch_bmnist_tied_bce-wl_gausk3 --is-tied True --pretrain-epochs 500 --finetune-epochs 300 --ae-mod-loss gausk3 --ae-main-loss bce-wl --alpha 9 &
sleep 10 &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT;
wait

echo "Work Finished!!"