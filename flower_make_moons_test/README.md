# Federated Learning, tests of a new approach to distributed learning on a toy model

This project aims to present the main properties of the Federated Learning (FL)
approach by testing it on a simple toy model.
Then, by modifying slightly the basic dataset, some more features are exploited
for future use on real problem.

## Contents

- [Federated Learning, tests of a new approach to distributed learning on a toy model](#federated-learning-tests-of-a-new-approach-to-distributed-learning-on-a-toy-model)
  - [Contents](#contents)
  - [Federated Learning (FL)](#federated-learning-fl)
    - [Federated Average (FedAvg)](#federated-average-fedavg)
  - [Dataset](#dataset)
  - [Model](#model)
  - [FL Framework](#fl-framework)
  - [Usage](#usage)
    - [Dependencies](#dependencies)
    - [General set up](#general-set-up)
    - [An example](#an-example)
  - [Analysis and results](#analysis-and-results)
  - [References](#references)

## Federated Learning (FL)

Federated Learning is a newly introduced approach to collaborative machine
learning.
There are nowadays many traditional machine learning algorithms which require
huge quantities of data raining examples to learn.
The rising problem is that is often very difficult to collect a sufficient
amount of data to reach a reasonable reliability.
This problem is very common in settings in which data access is restricted by
righteous privacy regulations, e.g. personal healthcare, or customization of
personal devices.
The implementation of a FL setting allows to avoid the necessity to collect
data in a single place.

A typical FL setting is built by taking into account a set of clients and a
centralized server.
The main steps of a FL iterative algorithm are the following:

1. global model initialization;
2. global model distribution;
3. local models training;
4. local models aggregation;
5. repeat from 2.

Firstly, step (1), the server, usually, initialize a unique global model to be
passed to the clients and keep the consistency of the procedure, i.e. ensuring
that every client runs the same learning algorithm.
Then, step (2), the server is in charge to distribute the initialized model to
all the available clients. FL algorithms allow to check how many clients are
available and train the same even if they are a low number. The distribution
procedure could take into account the numerosity of the single clients' datasets
(the only one information on the dataset coming from the clients).
After having received the global model, all the available clients train locally
the global model with their data, step (3), producing a new local model.
The weights of all the local models are then aggregated, step (4), at the server
place to be reduced to a new global model. The aggregation procedure, as the
distribution previously, could take into account the numerosity of the clients'
datasets.
Moreover the evaluation step, always performed locally by the clients, could be
aggregated to the server in an analogous procedure than local models. This allows
the server to have global perception of the performance of the learning step by
step. This action usually is inserted between step (2) and step (3).
The iterative procedure is finally repeated until the number of fixed federated
iterations is reached, usually.

### Federated Average (FedAvg)

Despite FL is a newly introduced approach, many aggregation procedures have been
proposed.
It is important to say that FL is not widely understood by now and because of this
not any aggregation procedure, or more generally FL algorithm, is reliable for any
specific problem.
The aggregation method used in this analysis is, probably, the most general: FedAvg.
This algorithm relies on Stochastic Gradient Descent (SGD) optimization method,
since the majority of the most successful deep learning works were based on this.
The available clients locally compute (step 3) their average gradient on their local
data at the current model ![$w_t$](https://latex.codecogs.com/svg.image?w_t),
where ![$t$](https://latex.codecogs.com/svg.image?t) identifies the federated round,
and the central server aggregates these gradients and applies the update

![\Large w_{t+1}\leftarrow w_t - \eta\sum_{k=1}^K\frac{n_k}{n}g_k](https://latex.codecogs.com/svg.image?w_%7Bt+1%7D%5Cleftarrow&space;w_t-%5Ceta%5Csum_%7Bk=1%7D%5EK%5Cfrac%7Bn_k%7D%7Bn%7Dg_k)

Above, ![$g_k=\nabla F_k(w_t)$](https://latex.codecogs.com/svg.image?g_k=\nabla&space;F_k(w_t)) is the average gradient of the client ![$k$](https://latex.codecogs.com/svg.image?k), ![$\eta$](https://latex.codecogs.com/svg.image?\eta) is
the learning rate, ![$n_k$](https://latex.codecogs.com/svg.image?n_k) is the number of samples at the client ![$k$](https://latex.codecogs.com/svg.image?k), ![$n$](https://latex.codecogs.com/svg.image?n) is the total
number of samples (sum over all the available clients).
Equivalently, the update can be given by

![$w_{t+1}\leftarrow w_t - \eta\sum_{k=1}^K\frac{n_k}{n}w_{t+1}^k$](https://latex.codecogs.com/svg.image?w_{t&plus;1}\leftarrow&space;w_t&space;-&space;\eta\sum_{k=1}^K\frac{n_k}{n}w_{t&plus;1}^k),

where ![$w_{t+1}^k\leftarrow w_t - \eta g_k$ $\forall k$](https://latex.codecogs.com/svg.image?w_{t&plus;1}^k\leftarrow&space;w_t&space;-&space;\eta&space;g_k$&space;$\forall&space;k).
In the last, every client takes a complete step of gradient descent, while the server
only takes the weighted average of the resulting models.

## Dataset

A simple toy dataset was chosen to set up a classification toy model to perform some
simulations in FL setting.
From the `scikit-learn` Python package, which provides a wide set of generators
for toy datsets, the `datasets.make_moons` generator was picked up.
This function produces the requested number of points in a 2-D space drawing two
interleaving circles, as the following figure shows.
![interleaving circles](images/make_moons_example.png?raw=true)
The same function returns also the classification array, that relates every point to
its corresponding circle.
One can make the request to add some noise to the generated points, and ask for the
points to be shuffled, once generated.
The noise value was fixed to 0.1 along every simulation.
It is also possible to set the random state that seeds the noise and shuffling, if
requested.
Two more functions, in addition to this settings, were built to transform a little
such generated dataset.
The first simply translates the dataset by a given vector
![$(dx, dy)$](https://latex.codecogs.com/svg.image?(dx,&space;dy)), i.e. every
point ![$(x, y)$](https://latex.codecogs.com/svg.image?(x,&space;y))
in the dataset undergoes the transformation
![$x'=x+dx$](https://latex.codecogs.com/svg.image?x%27=x+dx) and
![$y'=y+dy$](https://latex.codecogs.com/svg.image?y%27=y+dy).
The second applies a simple rotation by an angle
![$\theta$](https://latex.codecogs.com/svg.image?\theta) with a standard
transformation, i.e.
![$x'=x\cos(\theta)-y\sin(\theta)$](https://latex.codecogs.com/svg.image?x%27=x\cos(\theta)-y\sin(\theta))
and ![$y'=x\sin(\theta)-y\cos(\theta)$](https://latex.codecogs.com/svg.image?y%27=x\sin(\theta)+y\cos(\theta))
following the above notation.
Examples of translated and rotated datasets are shown in the following figure.

![dataset examples](images/datasets_examples.png?raw=true)

## Model

Every simulation was build on a simple two layers sequential model.
Both the layers are standard regular densely-connected Neural Network layers, the first
with 4 output, the second with 2, since the model is expected to classify the points
w.r.t. their circle of belonging.

![model grah](images/model_graph.png?raw=true)

## FL Framework

The framework adopted to make the necessary simulation is
[Flower: A Friendly Federated Learning Framework](https://flower.dev/).
The federated framework is set up by running a program for every client and one for
the server.
These will communicate using a RPC (Remote Procedure Call) framework exchanging the
weights of the model.
The ML framework used is `tensorflow` with a `keras.Sequential` model.
The client program accepts many parameters to build properly the client's dataset, and
to set up the outputs also.
The server program has a minimal configuration, receiving the number of clients in the
federated network and the number of the federated epochs to run.
Another simple program runs an aggregated model by building an equivalent aggregated
dataset.
Every client's dataset, as the aggregated one, is divided randomly in train set and test
set using the standard proportion 80-20, for train and test respectively.
The for every epoch an evaluation step is performed at every client's place as at the
aggregated model.
The performance of the model at every step, represented by the loss and the accuracy,
is retrieved and saved to be consulted later.
The loss function chosen is the Sparse Categorical Cross-Entropy, implemented by the
function `tensorflow.keras.losses.SparseCategoricalCrossentropy`.
The accuracy is simply computed as the ratio between the number of well classified points
and the total number of points in the test set.

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

Imagine you want to simulate in a FL set up 2 clients, with an entire dataset not
translated nor rotated, of 300 samples, for a number of 1000 federated epochs and
1 local epoch at each federated step. You want to use the default noise, the default
server and the you want to plot the decision boundary and get the learning curves.
Imagine you want to instatiate the server at the port 51550.
Imagine to have separate machines with different IP addresses.
The IP address of the server is 0.0.0.0, for example.
Ensure to have properly set the iptables and port forwarding, for a Linux system,
you can look at this [guide](https://www.systutorials.com/port-forwarding-using-iptables/)

In the server machine run

```bash
python3 server.py --rounds=1000 --n_clients=2 --address=0.0.0.0:51550
```

In the first client's machine run

```bash
python3 client.py --client_id=0 --n_samples=300 --n_clients=2 --plot=true --dump_curve=true --server=0.0.0.0:51550
```

In the second client's machine run

```bash
python3 client.py --client_id=1 --n_samples=300 --n_clients=2 --plot=true --dump_curve=true --server=0.0.0.0:51550
```

I order to run the aggregated one simply

```bash
python3 simple_model.py --n_clients=2 --n_samples=300 --n_epochs=1000 --plot=true
```

## Analysis and results

The easiest method to compare the performances of such models is to plot the losses and
the accuracies against the number of epochs (federated epochs for FL set ups, "standard"
epochs for the aggregated model).
Moreover, to have a general, and more informative, idea of the performance of the whole FL
set ups, the mean values of the losses and the accuracies of all the clients in every FL
set up are computed.
A confidence interval, that covers a confidence level of $68\%$, i.e. standard deviation
was taken, is then associated to these mean values, in order to give a more complete
representation.

The very first result to notice is that the aggregated model has a definitely faster convergence,
for both the loss and the accuracy, w.r.t. any other FL set up.
More the FL set up is distributed, slower is the convergence and higher is the standard
deviation between the clients.

![loss](images/loss_red_same.png?raw=true)
![accuracy](images/accuracy_red_same.png?raw=true)

The standard deviations are expected to tend to zero as the number of epochs increases.
Another useful comparison, using the final decision boundaries, underlines the fact that
with only 1000 federated epochs, the FL setting distributed between 8 clients is far from
the aggregated one.

![loss](images/db_single.png?raw=true)
![accuracy](images/db_distributed.png?raw=true)

The general setting changes when increasing the number of local epochs at clients' place.
The learning curves are here faster than before.
One can notice that somewhere in the plot, FL settings with an higher number of clients
outperforms others with a lower number of clients.

![loss](images/loss_red_adv.png?raw=true)
![accuracy](images/accuracy_red_adv.png?raw=true)

The last analysis, which tries to seek the TL ability of FL setting, shows that the set
ups which perform better are those which are more mixed up.
The simulation that used the rotation transformation is almost inconclusive, since the
convergence is not reach in many set ups.
However Fit is found the general behavior mentioned previously.

![loss](images/loss_red_TL2.png?raw=true)
![accuracy](images/accuracy_red_TL2.png?raw=true)

The simulation that used the translation transformation has many set ups, which nearly
reach the convergence, describe better the general behavior noticed.
The worst performances are those of the FL settings with 7 and 8 clients with transformed
datasets, then set ups with 1 and 2 clients with transformed datasets follow.

![loss](images/loss_red_TL1.png?raw=true)
![accuracy](images/accuracy_red_TL1.png?raw=true)

## References

1. Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D Flower: A Friendly Federated Learning Research Framework. arXiv preprint arXiv:2007.14390, 2020
2. M. Aledhari, R. Razzak, R. M. Parizi, and F. Saeed.  Federated learning:  A survey on enablingtechnologies, protocols, and applications.IEEE Access, 8:140699–140725, 2020
3. H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.Communication-efficient learning of deep networks from decentralized data, 2017
