## SIMULATION
These are the results of simulating 2 clients, 160 samples each, in a federated setting.
The datasets are generated using the make_moons method in the scikit-learn package, with
a fixed noise to 0.1. 
After being shuffled, the dataset are divided 80-20 in train-test sets. 
No cross validation has been used.

accuracy.png and loss.png contains the plot of the learning curves. In the legend of
these graphs there is the reference to the client to which they refer. With no_fed it
is referred the analogous non federated simulation, performed using the same model and
the same datasets (properly merged).

The data_*.png files show the drawings of the dataset for a visual comparison.

The *.dat files sgow the numbers from which the learning curves are plotted.

### LEARNING CURVES
![](loss.png?raw=true)
![](accuracy.png?raw=true)

### DATASETS
![](data_client_nofed.png?raw=true)
![](data_client_0.png?raw=true)
![](data_client_1.png?raw=true)
