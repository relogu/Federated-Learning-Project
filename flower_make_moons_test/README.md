# Federated Learning, tests of a new approach to distributed learning on a toy model

This project aims to present the main properties of the Federated Learning (FL)
approach by testing it on a simple toy model.
Then, by modifying slightly the basic dataset, some more features are exploited
for future use on real problem.

## Federated Learning (FL)

Federated Learning is a newly introduced approach to collaborative machine
learning[^1]
 \cite{9153560}.
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
This algorithm, firstly proposed in
\cite{mcmahan2017communicationefficient},
relies on Stochastic Gradient Descent (SGD) optimization method, since the majority
of the most successful deep learning works were based on this.
The available clients locally compute (step 3) their average gradient on their local
data at the current model $w_t$, where $t$ identifies the federated round, and the
central server aggregates these gradients and applies the update
$w_{t+1}\leftarrow w_t - \eta\sum_{k=1}^K\frac{n_k}{n}g_k$.
Above, $g_k=\nabla F_k(w_t)$ is the average gradient of the client $k$, $\eta$ is
the learning rate, $n_k$ is the number of samples at the client $k$, $n$ is the total
number of samples (sum over all the available clients).
Equivalently, the update can be given by
$w_{t+1}\leftarrow w_t - \eta\sum_{k=1}^K\frac{n_k}{n}w_{t+1}^k$,
where $w_{t+1}^k\leftarrow w_t - \eta g_k$ $\forall k$. In the last, every client takes
a complete step of gradient descent, while the server only takes the weighted average
of the resulting models. The following pseudo-code summarizes the procedure.

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
The first simply translates the dataset by a given vector $(dx, dy)$, i.e. every
point $(x, y)$ in the dataset undergoes the transformation $x'=x+dx$ and $y'=y+dy$.
The second applies a simple rotation by an angle $\theta$ with a standard transformation,
i.e. $x'=x\cos(\theta)-y\sin(\theta)$ and $y'=x\sin(\theta)-y\cos(\theta)$ following
the above notation.
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

Firstly set the server machine and launch the server program.

```bash
python3 server.py
```

The server starts waiting the handshake from clients to begin the federated iteration.
Parameters to pass are explain at

```bash
python3 server.py --help
usage: server.py [-h] --rounds ROUNDS --n_clients CLIENTS [--address ADDRES]

Server for moons test FL network using flower.
Give the nuber of federated rounds to pass to the strategy builder.
Give the minimum number of clients to wait for a federated averaging step.

optional arguments:
  -h, --help           show this help message and exit
  --rounds ROUNDS      number of federated rounds to perform
  --n_clients CLIENTS  minimum number of active clients to perform an iteration step
  --address ADDRESS     complete address to launch server, e.g. 127.0.0.1:8081
```

Then set a series of client machines and launch the client program for each machine.
The setting must be consistent following the parameters given.

```bash
python3 client.py
```

Parameters to pass are explain at

```bash
usage: client.py [-h] [--server SERVER] --client_id CLIENT_ID
                 [--rounds ROUNDS] --n_train N_TRAIN
                 [--noise NOISE] [--plot PLOT]

Client for moons test FL network using flower.
Give the id of the client and the number of local epoch you want to perform.
Give also the number of data samples you wanto to train for this client.
One can optionally give the server location to pass the client builder.
One can optionally give the noise to generate the dataset.
One can optionally tell the program to plot the decision boundary at the evaluation step.
The number of test data samples is fixed by the program.
The client id will also initialize the seed for the train dataset.
The program is built to make all the client use the same test dataset.

optional arguments:
  -h, --help            show this help message and exit
  --server SERVER       server address to point
  --client_id CLIENT_ID
                        client id, set also the seed for the dataset
  --rounds ROUNDS       number of local epochs to perform
  --n_train N_TRAIN     number of samples in training set
  --noise NOISE         noise to put in the train dataset
  --plot PLOT           tells the program whether to plot decision boundary or not
```

When the sufficient number of clients are connected an iteration step is performed.
After the number of iterations given is completed, server and clients return the final
performance.

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

[^1]: Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D Flower: A Friendly Federated Learning Research Framework. arXiv preprint arXiv:2007.14390, 2020
2. M. Aledhari, R. Razzak, R. M. Parizi, and F. Saeed.  Federated learning:  A survey on enablingtechnologies, protocols, and applications.IEEE Access, 8:140699–140725, 2020
3. H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.Communication-efficient learning of deep networks from decentralized data, 2017
