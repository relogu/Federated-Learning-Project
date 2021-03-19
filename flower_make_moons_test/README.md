# Test using FLOWER by Adap on sklearn.make_moons generated dataset
This test provides a client and a server program with minimal configuration

## Usage
Firstly set the server machine and launch the server program.
```bash
python3 server.py
```
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
Firstly set a series of client machines and launch the client program for each machine.
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

## Results
4 client (13,10,10,20)train_samples (1,1,1,1)local_rounds 5000fed_rounds 1000test_samples(shared)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--> (picking 2 out of 4 per federated round)  reached accuracy = 1 before end\
4 client (13,10,10,20)train_samples (3,2,2,5)local_rounds 5000fed_rounds 1000test_samples(shared)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--> (picking 2 out of 4 per federated round)  reached accuracy = 0.9670000076293945\
4 client (13,10,10,20)train_samples (1,1,1,1)local_rounds 5000fed_rounds 1000test_samples(shared)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--> loss: 0.1043 - accuracy: 1.0000\
4 client (13,10,10,20)train_samples (5,5,5,5)local_rounds 5000fed_rounds 1000test_samples(shared)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--> loss: 0.0505 - accuracy: 0.9820\
4 client (13,10,10,20)train_samples (3,2,2,5)local_rounds 5000fed_rounds 1000test_samples(shared)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--> loss: 0.0937 - accuracy: 0.9670  


## References

<blockquote>1- Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D Flower: A Friendly Federated Learning Research Framework. arXiv preprint arXiv:2007.14390, 2020. </blockquote>