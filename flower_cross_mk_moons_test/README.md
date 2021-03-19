# Test using FLOWER by Adap with cross-validation on sklearn.make_moons generated dataset
This test provides a client and a server program with minimal configuration

## Usage
Firstly set the server machine and launch the server program.
```bash
python3 server.py
```
Parameters to pass are explain at
```bash
python3 server.py -h
usage: server.py [-h] --rounds ROUNDS --n_clients CLIENTS [--address ADDRESS]

Server for moons test FL network using flower.
Give the nuber of federated rounds to pass to the strategy builder.
Give the minimum number of clients to wait for a federated averaging step.

optional arguments:
  -h, --help           show this help message and exit
  --rounds ROUNDS      number of federated rounds to perform
  --n_clients CLIENTS  minimum number of active clients to perform an iteration step
  --address ADDRESS    complete address to launch server, e.g. 127.0.0.1:8081
```
Firstly set a series of client machines and launch the client program for each machine.
```bash
python3 client.py
```
Parameters to pass are explain at
```bash
python3 client.py -h
2021-03-19 14:41:15.896331: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
usage: client.py [-h] [--server SERVER] --client_id CLIENT_ID [--rounds ROUNDS] --n_samples N_SAMPLES [--n_folds N_FOLDS] [--loss LOSS] [--plot PLOT]

Client for moons test FL network using flower and cross-validation.
Give the id of the client and the number of local epoch you want to perform.
Give also the number of data samples for this client.
One can optionally give the number of folds to use for the cross validation.
One can optionally give the server location to pass the client builder.
One can optionally tell the program to plot the decision boundary at the evaluation step (only for the chosen model).
The client id will also initialize the seed for the train dataset.

optional arguments:
  -h, --help            show this help message and exit
  --server SERVER       server address to point
  --client_id CLIENT_ID
                        client id, set also the seed for the dataset
  --rounds ROUNDS       number of local epochs to perform
  --n_samples N_SAMPLES
                        number of total samples
  --n_folds N_FOLDS     number of folds for the cross validator
  --loss LOSS           tells the program whether to use loss or accuracy for model selection
  --plot PLOT           tells the program whether to plot decision boundary or not
```

## Results



## References

<blockquote>1- Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D Flower: A Friendly Federated Learning Research Framework. arXiv preprint arXiv:2007.14390, 2020. </blockquote>