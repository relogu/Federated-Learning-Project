# Federated Learning, tests of different clustering algorithms in a federated setting

This project aims to present the application of the Federated Learning (FL) approach to some clustering algorithm, in order to identify the clusters inside the dataset.

## Contents

- [Federated Learning, tests of different clustering algorithms in a federated setting](#federated-learning-tests-of-different-clustering-algorithms-in-a-federated-setting)
  - [Contents](#contents)
  - [Federated Learning (FL)](#federated-learning-fl)
  - [Datasets used](#datasets-used)
    - [Blobs](#blobs)
    - [Moons](#moons)
    - [MNIST](#mnist)
  - [Models and algorithms](#models-and-algorithms)
  - [FL Framework](#fl-framework)
  - [Usage](#usage)
    - [Dependencies](#dependencies)
    - [General set up](#general-set-up)
    - [An example](#an-example)
  - [Analysis and results](#analysis-and-results)
  - [References](#references)

## Federated Learning (FL)

For a brief introduction to FL and its main concepts look at
[this](https://github.com/relogu/Federated-Learning-Project/tree/master/flower_make_moons_test#federated-learning-fl)
section of the
[README.md](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/README.md)
file of the
[this](https://github.com/relogu/Federated-Learning-Project/tree/master/flower_make_moons_test)
repository.

## Datasets used

TODO

### Blobs

TODO

### Moons

TODO

### MNIST

TODO

## Models and algorithms

TODO

## FL Framework

The framework adopted to make the necessary simulation is
[Flower: A Friendly Federated Learning Framework](https://flower.dev/).
The federated framework is set up by running a program for every client and one for
the server.
These will communicate using a RPC (Remote Procedure Call) framework exchanging the
weights of the model.

## Usage

In order to perfom simulations using these programs, a virtual environment is preferred.
Alternatively one can use any OS that is able o run python.
These programs have been tested on a [conda](https://docs.conda.io/en/latest/) virtual
environment on Linux.

### Dependencies

There are few dependencies that can be managed with any package manager.
The mandatory packages can be installed using `pip3` using the command

```bash
pip3 install flwr tensorflow scikit-learn matplotlib numpy
```

### General set up

Firstly set the server machine and launch the server program:
[server.py](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/server.py).

```bash
python3 server.py
```

The server starts waiting the handshake from clients to begin the federated iteration.
Parameters to pass are explain at

```bash
```

Then set a series of client machines and launch the client program for each machine.
The client program is given in
[client.py](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/client.py).
The setting must be consistent following the parameters given.

```bash
python3 client.py
```

Parameters to pass are explain at

```bash
```

When the sufficient number of clients are connected an iteration step is performed.
After the number of iterations given is completed, server and clients return the final
performance.

The code inside
[common_fn.py](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/common_fn.py)
is necessary, because contains some of the functions used by the procedures above.

Additionally, inside the folder
[test](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/test)
are provided the necessary files to perform the tests on the functions.
I fyou want to run test simply go with

```bash
python3 function_tests.py
```

in the [test](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/test) folder

The folder
[scripts](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/scripts)
provides the scripts used for plotting some of the images inside the [Analysis and results](#analysis-and-results) part

### An example

TODO

Imagine you want to instatiate the server at the port 51550.
Imagine to have separate machines with different IP addresses.
The IP address of the server is 0.0.0.0, for example.
Ensure to have properly set the iptables and port forwarding, for a Linux system,
you can look at this [guide](https://www.systutorials.com/port-forwarding-using-iptables/)

Some simulations could be executed by launching
[run.sh](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/run.sh)
script (working on a Ubuntu 20.04 satisfying the dependencies listed above)

## Analysis and results

In [images](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/images) folder are provided some output figures representing the resuslts of the simulations in [run.sh](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/run.sh).
Every folder in [images](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/images) represents one of the simulations whose results are presented in the correspondent README file.

## References

1. Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D Flower: A Friendly Federated Learning Research Framework. arXiv preprint arXiv:2007.14390, 2020
2. M. Aledhari, R. Razzak, R. M. Parizi, and F. Saeed.  Federated learning:  A survey on enabling technologies, protocols, and applications.IEEE Access, 8:140699–140725, 2020
3. H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.Communication-efficient learning of deep networks from decentralized data, 2017
