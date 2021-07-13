# Federated Learning, tests of different clustering algorithms in a federated setting

This project aims to present the application of the Federated Learning (FL) approach to some clustering algorithm, in order to identify the clusters inside the dataset.

## Contents

- [Federated Learning, tests of different clustering algorithms in a federated setting](#federated-learning-tests-of-different-clustering-algorithms-in-a-federated-setting)
  - [Contents](#contents)
  - [Federated Learning (FL)](#federated-learning-fl)
  - [Datasets](#datasets)
    - [Blobs](#blobs)
    - [Moons](#moons)
    - [MNIST](#mnist)
  - [EUROMDS](#euromds)
  - [Models and algorithms](#models-and-algorithms)
    - [Simple k-means](#simple-k-means)
    - [Simplified k-FED](#simplified-k-fed)
    - [Unsupervised deep embedding for clustering (UDE)](#unsupervised-deep-embedding-for-clustering-ude)
    - [ClusterGAN](#clustergan)
  - [FL Framework](#fl-framework)
  - [Usage](#usage)
    - [Dependencies](#dependencies)
    - [General set up](#general-set-up)
    - [An example](#an-example)
  - [Analysis and results](#analysis-and-results)
    - [Metrics](#metrics)
  - [References](#references)

## Federated Learning (FL)

For a brief introduction to FL and its main concepts look at
[this](https://github.com/relogu/Federated-Learning-Project/tree/master/flower_make_moons_test#federated-learning-fl)
section of the
[README.md](https://github.com/relogu/Federated-Learning-Project/blob/master/flower_make_moons_test/README.md)
file of the
[this](https://github.com/relogu/Federated-Learning-Project/tree/master/flower_make_moons_test)
repository.

## Datasets

Different dataset were used to test the algorithm in [Models and algorithms](#models-and-algorithms).
Tuning the configuration of the clients, the user is able to create imbalanced non-iid partitions of the selected dataset using Latent Dirichlet Allocation (LDA) without resampling.
This procedure could be useful to simulate real distributions of data.
The size of the entire dataset is tunable, as the number of clients, but it is recomended to take into account the partitioning while setting these values.
Every partioned dataset is subdivided in train and test, respectively the 80% and the 20% of the whole dataset.

### Blobs

This dataset is generated exploiting [this](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs) method of [scikit-learn](https://scikit-learn.org/stable/index.html).
As the description says, a defined number of isotropic Gaussian blobs are generated for clustering; the labels are provided also.
In this specific case the user is able to selected the number of clusters manually, while the number of dimensions of the space is fixed to 30.
The random state is tunable in the program.

### Moons

This dataset is generated exploiting [this](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons) method of [scikit-learn](https://scikit-learn.org/stable/index.html).
As described, two interleaving half circles are generated; the labels are provided also.
Clearly here the number of clusters is fixed to 2 as the number of dimensions of the data space, 2 also.
The random state is tunable in the program.

### MNIST

This is the old-fashioned but widely used handwritten digits dataset.
For those who are not familiar with this data, [this page](https://it.wikipedia.org/wiki/MNIST_database) provide many useful information.

## EUROMDS

TODO

## Models and algorithms

Many different models and algorithms are implemented in order to perform th unsupervised clustering on the preferred dataset.
Many others are in working.
The objective is to have enough results to allow the best choice of the algorithm for every dataset.

### Simple k-means

The implementation from [scikit-learn.cluster](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html#sklearn.cluster.k_means) is used in the federated setting initializing the algorithm at every client with the *k-means++* initilization and then et every federated step the centroids of the clients are averaged to initialize the next local step of k-means.
For a deeper explanation on how the k-means algorithm works look at [this](https://en.wikipedia.org/wiki/K-means_clustering) page.

### Simplified k-FED

This new algorithm has been proposed in [3].
It is based on a strict definition of heterogeneity of the distributed data.
The implementation in this work exploits simple k-means as baseline algorithm and then following the same reasoning of [3] to built the centroids of clusters.

### Unsupervised deep embedding for clustering (UDE)

The implementation of this model is based on the paper [4].
It exploits the use of an autoencoder (AE), a Neural Network (NN) that tries to map the data into themselves, to make a dimensionality reduction.
Once the autoencoder has been pre-trained, a clustering algorithm is used to initialize the weights of a statistical clustering layer that performs the final embedding.
In this implementation the clustering algorithm exploited is simplified k-FED.
The clustering layer, that is build on the top of the encoder part of the AE, tries to maximize the clustering label probability exploiting the t-Stundet's distribution.

### ClusterGAN

ClusterGAN is a mechanism for clustering using Generative Adversarial networks (GANs) introduced in [5].
By sampling latent variables from a mixture of one-hot encoded variables and continuous latent variables, coupled with an inverse network (which projects the data to the latent space) trained jointly with a clustering specific loss, we are able to achieve clustering in the latent space.
In [5] is shown a remarkable phenomenon that GANs can preserve latent space interpolation across categories, even though the discriminator is never exposed to such vectors.
Until now this model can only be used to cluster the MNIST dataset, for the others the implemetation is a work in progress.

## FL Framework

The framework adopted to make the necessary simulation is [Flower: A Friendly Federated Learning Framework](https://flower.dev/), presented in [1] .
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

If one wants to clone the conda environment used for these simulations, [spec-file.txt](https://github.com/relogu/Federated-Learning-Project/blob/master/spec-file.txt) provides the list of packages necessary for the program.
To create directly the conda environment one can simply use

```bash
conda create --name <env> --file spec-file.txt
```

After having cloned the environment it's howevere necessary to install *flwr* with *pip*, because it is not provided in conda.

### General set up

Firstly set the server machine and launch the server program:
[server.py](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/server.py).

```bash
python3 server.py
```

The server starts waiting the handshake from clients to begin the federated iteration.
Parameters to pass are explain at

```bash
usage: server.py [-h] --n_clients {2,3,4,5,6,7,8} [--strategy {fed_avg,k-fed,fed_avg_k-means,clustergan}] [--ae_epochs AE_EPOCHS] [--kmeans_epochs KMEANS_EPOCHS]
                 [--cluster_epochs CLUSTER_EPOCHS] [--total_epochs TOTAL_EPOCHS] [--address ADDRESS]

Server program for moons test FL network using flower.
Give the number of federated rounds to pass to the strategy builder.
Give the minimum number of clients to wait for a federated averaging step.
Give optionally the complete address onto which instantiate the server.

optional arguments:
  -h, --help            show this help message and exit
  --n_clients {2,3,4,5,6,7,8}
                        minimum number of active clients to perform an iteration step
  --strategy {fed_avg,k-fed,fed_avg_k-means,clustergan}
                        strategy for the server
  --ae_epochs AE_EPOCHS
                        number of federated epoch to preform the autoencoder step
  --kmeans_epochs KMEANS_EPOCHS
                        number of federated epoch to preform the k-means step
  --cluster_epochs CLUSTER_EPOCHS
                        number of federated epoch to preform the clustering step
  --total_epochs TOTAL_EPOCHS
                        number of total federated epochs to perform, used in clustergan strategy
  --address ADDRESS     complete address to launch server, e.g. 127.0.0.1:8081
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
usage: client.py [-h] --client_id CLIENT_ID --dataset {blobs,moons,mnist} --alg {k-means,k_fed-ae_clust,k-ae_clust,clustergan} --n_samples N_SAMPLES --n_clients
                 {2,3,4,5,6,7,8} --n_clusters N_CLUSTERS [--server SERVER] [--rounds ROUNDS] [--seed SEED] [--noise NOISE] [--lda LDA]

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
  --dataset {blobs,moons,mnist}
                        client dataset identifier
  --alg {k-means,k_fed-ae_clust,k-ae_clust,clustergan}
                        algorithm identifier
  --n_samples N_SAMPLES
                        number of total samples in whole training set
  --n_clients {2,3,4,5,6,7,8}
                        number of total clients in the FL setting
  --n_clusters N_CLUSTERS
                        number of total clusters to initialize the kMeans algorithm
  --server SERVER       server address to point
  --rounds ROUNDS       number of local epochs to perform at each federated epoch
  --seed SEED           set the seed for the random generator of the whole dataset
  --noise NOISE         noise to put in the train dataset
  --lda LDA             wheater to apply LDA partitioning to the entire dataset
```

When the sufficient number of clients are connected an iteration step is performed.
After the number of iterations given is completed, server and clients return the final performance.

The code inside [common_fn.py](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/py/common_fn.py) is necessary, because contains some of the functions used by the procedures above.

The code inside [clustergan.py](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/py/clustergan.py) is necessary, because contains all the necessary to build the ClusterGAN model.
It can be also used as a stand-alone model to test the centralized ClusterGAN.

```bash
usage: clustergan.py [-h] [-n N_EPOCHS] [-b BATCH_SIZE] [-i IMG_SIZE] [-d LATENT_DIM] [-l LEARNING_RATE] [-c N_CRITIC] [-w]

ClusterGAN Training Script

optional arguments:
  -h, --help            show this help message and exit
  -n N_EPOCHS, --n_epochs N_EPOCHS
                        Number of epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -i IMG_SIZE, --img_size IMG_SIZE
                        Size of image dimension
  -d LATENT_DIM, --latent_dim LATENT_DIM
                        Dimension of latent space
  -l LEARNING_RATE, --lr LEARNING_RATE
                        Learning rate
  -c N_CRITIC, --n_critic N_CRITIC
                        Number of training steps for discriminator per iter
  -w, --wass_flag       Flag for Wasserstein metric
```

Additionally, inside the folder [test](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/test) are provided the necessary files to perform the tests on the functions.
If you want to run test simply go with

```bash
python3 function_tests.py
```

in the [test](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/test) folder

The folder [scripts](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/scripts) provides the scripts used for plotting some of the images inside the [Analysis and results](#analysis-and-results) part

### An example

Imagine you want to instatiate the server at the port 51550.
Imagine to have separate machines with different IP addresses.
The IP address of the server is 0.0.0.0, for example.
Ensure to have properly set the iptables and port forwarding, for a Linux system,
you can look at this [guide](https://www.systutorials.com/port-forwarding-using-iptables/)

Some simulations could be executed by launching
[run.sh](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/run.sh)
script (working on a Ubuntu 20.04 satisfying the dependencies listed above)

## Analysis and results

In [results](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results) folder are provided some output figures representing the results of the simulations in [run.sh](https://github.com/relogu/Federated-Learning-Project/blob/master/clustering/run.sh).
Every folder in [results](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results) represents one of the simulations whose results are presented in the correspondent README file.

| Dataset | LDA partitioning | Model | Results |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Blobs|yes|k-means|[README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/lda_k-means)|
|Blobs|no|k-means|[README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/k-means)|
|Blobs|yes|k-FED + UDE|[README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/lda_k-fed_ae)|
|Blobs|no|k-FED + UDE|[README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/k-fed_ae)|
|MNIST|yes|ClusterGAN|TODO|
|MNIST|no|ClusterGAN|TODO|
|EUROMDS|no|ClusterGAN|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDS_clustergan5k)<br/>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDS_clustergan10k)|
|EUROMDSr|no|ClusterGAN|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSr_clustergan5k)<br/>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSr_clustergan10k)|
|EUROMDSrr|no|ClusterGAN|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrr_clustergan5k)<br>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrr_clustergan10k)|
|EUROMDSrrr|no|ClusterGAN|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrrr_clustergan5k)<br>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrrr_clustergan10k)|
|EUROMDSrrrr|no|ClusterGAN|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrrrr_clustergan5k)<br>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrrrr_clustergan10k)|
|EUROMDS|no|k-means + UDE|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDS_ude1k5d2k)<br/>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDS_ude2k1u4k)|
|EUROMDSr|no|k-means + UDE|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSr_ude1k5d2k)<br/>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSr_ude2k1u4k)|
|EUROMDSrr|no|k-means + UDE|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrr_ude1k5d2k)<br>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrr_ude2k1u4k)|
|EUROMDSrrr|no|k-means + UDE|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrrr_ude1k5d2k)<br>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrrr_ude2k1u4k)|
|EUROMDSrrrr|no|k-means + UDE|5k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrrrr_ude1k5d2k)<br>10k epochs: [README.md](https://github.com/relogu/Federated-Learning-Project/tree/master/clustering/results/EUROMDSrrrr_ude2k1u4k)|

### Metrics

The metrics used are the following

- accuracy, simply the clustering accuracy as the ratio of well classified sample on the total number of samples;
- adjusted mutual information score (AMI), it is used the [implementation in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score);
- adjusted random score (ARI), it is used the [implementation in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score);
- homogeneity scoro (HOMO), it is used the [implementation in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score);
- normalized mutual information score (NMI), it is used the [implementation in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score);
- random score (RAN), it is used the [implementation in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html#sklearn.metrics.rand_score).

## References

1. Daniel J. Beutel and Taner Topal and Akhil Mathur and Xinchi Qiu and Titouan Parcollet and Nicholas D. Lane (2020). Flower: A Friendly Federated Learning Research Framework. CoRR, abs/2007.14390.
2. H. Brendan McMahan and Eider Moore and Daniel Ramage and Blaise Agüera y Arcas (2016). Federated Learning of Deep Networks using Model Averaging. CoRR, abs/1602.05629.
3. Don Kurian Dennis and Tian Li and Virginia Smith (2021). Heterogeneity for the Win: One-Shot Federated Clustering. CoRR, abs/2103.00697.
4. Junyuan Xie and Ross B. Girshick and Ali Farhadi (2015). Unsupervised Deep Embedding for Clustering Analysis. CoRR, abs/1511.06335.
5. Sudipto Mukherjee and Himanshu Asnani and Eugene Lin and Sreeram Kannan (2018). ClusterGAN : Latent Space Clustering in Generative Adversarial Networks. CoRR, abs/1809.03627.
