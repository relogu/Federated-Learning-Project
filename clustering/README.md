# Federated Learning, tests of different clustering algorithms ina federated setting

This project aims to present the application of the Federated Learning (FL) approach to some clustering algorithm, in order to identify the clusters inside the dataset.

## Contents

- [Federated Learning, tests of different clustering algorithms ina federated setting](#federated-learning-tests-of-different-clustering-algorithms-ina-federated-setting)
  - [Contents](#contents)
  - [Federated Learning (FL)](#federated-learning-fl)
  - [Datasets used](#datasets-used)
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


## Models and algorithms

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
[server.py](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/server.py).

```bash
python3 server.py
```

The server starts waiting the handshake from clients to begin the federated iteration.
Parameters to pass are explain at

```bash
usage: server.py [-h] --rounds ROUNDS --n_clients CLIENTS [--address ADDRESS]

Server program for moons test FL network using flower.
Give the number of federated rounds to pass to the strategy builder.
Give the minimum number of clients to wait for a federated averaging step.
Give optionally the complete address onto which instantiate the server.

optional arguments:
  -h, --help           show this help message and exit
  --rounds ROUNDS      number of federated rounds to perform
  --n_clients CLIENTS  minimum number of active clients to perform an iteration step
  --address ADDRESS    complete address to launch server, e.g. 127.0.0.1:8081
```

Then set a series of client machines and launch the client program for each machine.
The client program is given in
[client.py](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/client.py).
The setting must be consistent following the parameters given.

```bash
python3 client.py
```

Parameters to pass are explain at

```bash
usage: client.py [-h] --client_id CLIENT_ID --n_samples N_SAMPLES --n_clients N_CLIENTS [--server SERVER] [--rounds ROUNDS] [--noise NOISE] [--is_rotated IS_ROTATED]
                 [--is_traslated IS_TRASLATED] [--test TEST] [--plot PLOT] [--dump_curve L_CURVE]

Client for moons test FL network using flower.
Give the id of the client and the number of local epoch you want to perform.
Give also the number of data samples you want to train for this client.
Give also the number of clients in the FL set up to build properly the dataset.
One can optionally give the server location to pass the client builder.
One can optionally give the number of local epochs to perform.
One can optionally give the noise to generate the dataset.
One can optionally tell the program to plot the decision boundary at the evaluation step.
One can optionally tell the program to use the shared test set (default) or the train set as test also.
The client id will also initialize the seed for the train dataset.

optional arguments:
  -h, --help            show this help message and exit
  --client_id CLIENT_ID
                        client identifier
  --n_samples N_SAMPLES
                        number of total samples in whole training set
  --n_clients N_CLIENTS
                        number of total clients in the FL setting
  --server SERVER       server address to point
  --rounds ROUNDS       number of local epochs to perform at each federated epoch
  --seed SEED           set the seed for the random generator of the whole dataset
  --noise NOISE         noise to put in the train dataset
  --is_rotated IS_ROTATED
                        set true for producing a rotated dataset
  --is_traslated IS_TRASLATED
                        set true for producing a traslated dataset
  --test TEST           tells the program whether to use the shared test dataset (True) or the train dataset as test (False)
  --plot PLOT           tells the program whether to plot decision boundary, every 100 federated epochs, or not
  --dump_curve L_CURVE  tells the program whether to dump the learning curve, at every federated epoch, or not
```

When the sufficient number of clients are connected an iteration step is performed.
After the number of iterations given is completed, server and clients return the final
performance.

Another short program is provided,
[simple_model.py](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/simple_model.py),
to simulate an aggregated version of the model.

```bash
python3 simple_model.py
```

Parameters to pass are explain at

```bash
usage: simple_model.py [-h] --n_clients N_CLIENTS --n_samples N_SAMPLES --n_epochs N_EPOCHS [--is_traslated IS_TRASLATED] [--is_rotated IS_ROTATED] [--noise NOISE] [--plot PLOT]

Simple model to compare results of FL approach.
It simulates, once properly set, the same set up of a FL distribution in an aggregated version.
The learning curve is dumped at every epoch.

optional arguments:
  -h, --help            show this help message and exit
  --n_clients N_CLIENTS
                        maximum number of different clients to simulate, used to create the dataset
  --n_samples N_SAMPLES
                        number of total samples
  --n_epochs N_EPOCHS   number of total epochs for the training
  --is_traslated IS_TRASLATED
                        set true in the case of a traslated datset
  --is_rotated IS_ROTATED
                        set true in the case of a rotated datset
  --noise NOISE         noise to add to dataset
  --plot PLOT           tells the program whether to plot decision boundary, every 100 epochs, or not
```

The code inside
[common_fn.py](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/common_fn.py)
is necessary, because contains some of the functions used by the procedures above.

Additionally, inside the folder
[test](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/test)
are provided the necessary files to perform the tests on the functions.
I fyou want to run test simply go with

```bash
python3 function_tests.py
```

in the [test](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/test) folder

The folder
[scripts](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/scripts)
provides the scripts used for plotting some of the images inside the [Analysis and results](#analysis-and-results) part

### An example

TODO

Imagine you want to instatiate the server at the port 51550.
Imagine to have separate machines with different IP addresses.
The IP address of the server is 0.0.0.0, for example.
Ensure to have properly set the iptables and port forwarding, for a Linux system,
you can look at this [guide](https://www.systutorials.com/port-forwarding-using-iptables/)

Some simulations could be executed by launching 
[run.sh]()
script (working on a Ubuntu 20.04 satisfying the dependencies listed above)

## Analysis and results

In [images]() folder are provided some output figures representing the resuslts of the simulations in [run.sh]().
Every folder in [images]() represents one of the simulations whose results are presented in the correspondent README file.

## References

1. Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D Flower: A Friendly Federated Learning Research Framework. arXiv preprint arXiv:2007.14390, 2020
2. M. Aledhari, R. Razzak, R. M. Parizi, and F. Saeed.  Federated learning:  A survey on enabling technologies, protocols, and applications.IEEE Access, 8:140699–140725, 2020
3. H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.Communication-efficient learning of deep networks from decentralized data, 2017
