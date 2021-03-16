# Test using FLOWER by Adap and sklearn.make_moons
This test provides a client and a server program with minimal configuration

## Usage
Firstly set the server machine and launch the server program.
```bash
python3 server.py
```
Parameters to pass are explain at
```bash
python3 server.py --help
```
Firstly set a series of client machines and launch the client program for each machine.
```bash
python3 client.py
```
Parameters to pass are explain at
```bash
python3 client.py --help
```

## Results
4 client (13,9,10,20)train_samples (1,1,1,1)local_rounds 10000fed_rounds 1000test_samples(shared) --> loss=0.14580243825912476; accuracy=0.369999766349792
4 client (13,9,10,20)train_samples (1,1,1,1)local_rounds 1000fed_rounds 1000test_samples(shared) --> loss=0.26961350440979004; accuracy=0.9129999876022339
4 client (13,9,11,20)train_samples (3,2,3,5)local_rounds 1000fed_rounds 1000test_samples(shared) --> loss=0.29000794887542725; accuracy=0.8820000290870667


## References

<blockquote>1- Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D Flower: A Friendly Federated Learning Research Framework. arXiv preprint arXiv:2007.14390, 2020. </blockquote>
```