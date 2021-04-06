# SIMULATION
These are the results of simulating 4 clients, 210 samples each, in a federated setting. 
The datasets are generated using the make_moons method in the scikit-learn package, with a fixed noise to 0.1.
After being shuffled, the dataset are divided 80-20 in train-test sets. 
No cross validation has been used.

The files accuracy.png and loss.png contains the plot of the learning curves.
In the legend of these graphs is written the client to which they refer, the mean value is also plotted usign a confidence interval estimated by the standard deviation.

The data_*.png files show the drawings of the dataset for a visual comparison.

The dec_bound_*.png files show the drawings of the final decision boundaries obtained by the model on the train set of the client.

The *.dat files show the numbers from which the learning curves are plotted.

### LEARNING CURVES
![](loss.png?raw=true)
![](accuracy.png?raw=true)

### DATASETS
![](data_client_0.png?raw=true)
![](data_client_1.png?raw=true)
![](data_client_2.png?raw=true)
![](data_client_3.png?raw=true)

### DECISION BOUNDARIES
![](dec_bound_c0.png?raw=true)
![](dec_bound_c1.png?raw=true)
![](dec_bound_c2.png?raw=true)
![](dec_bound_c3.png?raw=true)