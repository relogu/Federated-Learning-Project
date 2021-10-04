#!/bin/bash

echo $PWD
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "Number of clusters $1"
CLUSTERS="$1"
echo "Output folder $2"
OUT_FOL="$2"
echo "Fill NaNs $3"
FILL=""
DS="final"
if [ $3 = "fill" ]
then
FILL="--fill"
DS="fill"
fi
echo "Dataset type chosen $DS"
echo "Use orthogonality constraint $4"
ORTHO=""
if [ $4 = "y" ]
then
ORTHO="--ortho"
fi

DATASET="--groups Genetics --groups CNA --ex_col UTX --ex_col CSF3R --ex_col SETBP1 --ex_col PPM1D"

mkdir "$PWD/$OUT_FOL"
# entire dataset
python3 py/udec/main.py $FILL $ORTHO --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.20 --ran_flip 0.20 --tied $DATASET --folder "$PWD/$OUT_FOL" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/DEC_EUROMDS{$DS}_single_a{$ORTHO}_K$CLUSTERS"
mv "$PWD/$OUT_FOL"/* "$PWD/results/DEC_EUROMDS{$DS}_single_a{$ORTHO}_K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# entire dataset
python3 py/udec/main.py $FILL $ORTHO --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.05 --ran_flip 0.05 --tied $DATASET --folder "$PWD/$OUT_FOL" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/DEC_EUROMDS{$DS}_single_b{$ORTHO}_K$CLUSTERS"
mv "$PWD/$OUT_FOL"/* "$PWD/results/DEC_EUROMDS{$DS}_single_b{$ORTHO}_K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# entire dataset
python3 py/udec/main.py $FILL $ORTHO --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.20 --ran_flip 0.20 --tied $DATASET --folder "$PWD/$OUT_FOL" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/DEC_EUROMDS{$DS}_single_c{$ORTHO}_K$CLUSTERS"
mv "$PWD/$OUT_FOL"/* "$PWD/results/DEC_EUROMDS{$DS}_single_c{$ORTHO}_K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# entire dataset
python3 py/udec/main.py $FILL $ORTHO --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.10 --ran_flip 0.10 --tied $DATASET --folder "$PWD/$OUT_FOL" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/DEC_EUROMDS{$DS}_single_d{$ORTHO}_K$CLUSTERS"
mv "$PWD/$OUT_FOL"/* "$PWD/results/DEC_EUROMDS{$DS}_single_d{$ORTHO}_K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# entire dataset
python3 py/udec/main.py $FILL $ORTHO --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.05 --ran_flip 0.05 --tied $DATASET --folder "$PWD/$OUT_FOL" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/DEC_EUROMDS{$DS}_single_e{$ORTHO}_K$CLUSTERS"
mv "$PWD/$OUT_FOL"/* "$PWD/results/DEC_EUROMDS{$DS}_single_e{$ORTHO}_K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# entire dataset
python3 py/udec/main.py $FILL $ORTHO --u_norm --ae_epochs 2500 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.01 --ran_flip 0.01 --tied $DATASET --folder "$PWD/$OUT_FOL" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/DEC_EUROMDS{$DS}_single_f{$ORTHO}_K$CLUSTERS"
mv "$PWD/$OUT_FOL"/* "$PWD/results/DEC_EUROMDS{$DS}_single_f{$ORTHO}_K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"

mkdir "$PWD/$OUT_FOL"
# entire dataset
python3 py/udec/main.py $FILL $ORTHO --u_norm --ae_epochs 5000 --cl_epochs 20000 --n_clusters $CLUSTERS --dropout 0.01 --ran_flip 0.01 --tied $DATASET --folder "$PWD/$OUT_FOL" --hardware_acc
wait
sleep 5
mkdir "$PWD/results/DEC_EUROMDS{$DS}_single_g{$ORTHO}_K$CLUSTERS"
mv "$PWD/$OUT_FOL"/* "$PWD/results/DEC_EUROMDS{$DS}_single_g{$ORTHO}_K$CLUSTERS"/
sleep 10
rmdir "$PWD/$OUT_FOL"
