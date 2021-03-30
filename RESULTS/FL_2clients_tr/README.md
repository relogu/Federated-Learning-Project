## SIMULATION
These are the results of simulating 2 clients, 160 samples each, in a federated setting. 
The datasets are generated using the make_moons method in the scikit-learn package, with
a fixed noise to 0.1. 
In addition every dataset is randomly traslated horizontally by a value dx&isin;[-0.1 ; 0.1] 
and vertically by a value dy&isin;[-0.1 ; 0.1]. 
After being shuffled, the dataset are divided 80-20 in train-test sets.
No cross validation has been used.

accuracy.png and loss.png contains the plot of the learning curves. In the legend of
these graphs there is the reference to the client to which they refer. With no_fed it
is referred the analogous non federated simulation, performed using the same model and
the same datasets (properly merged).

The data_*.png files show the drawings of the dataset for a visual comparison.

The *.dat files show the numbers from which the learning curves are plotted.