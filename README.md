# Federated Learning Project

This repository contains all the folders and material used to prepare the project on FL.

## How to use this repo

### Create the environment

In order to create the same exact environment that was used durong the experimentations, some .yml files are provided.
Espe.cially,
TODO(torch-gpu.yml)[]
file contains all the necessary packages for running this repo.
Using the following command, ``conda`` will create an environment named "torch-gpu" from the .yml file.

``bash
conda env create -f environment.yml

``

It might be necessary to modify the version of the CUDA libreries. In this case refer to the official guides for PyTorch and CUDA.
In the situation in which you don't want to use accelerations, it is possible to get only the package for PyTorch with CPU. Refer to the official guide in this case also.
It is possible that ``tensorflow_federated`` package won't be installed using this procedure. Refer to the official guide to install it (especially looking carefully at its compatibility with the CUDA libraries).

In order to use the MNIST baselines and their federated versions, you may want to look at the (datasets)[] folder.
TODO

### Add the path to pythonpath
TODO

export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
