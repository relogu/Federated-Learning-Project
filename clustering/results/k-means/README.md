# k-means

In this simulation the usual k-means algorithm for clustering is used on the Blobs dataset.
The following graphs represents the metrics used to evaluate the performance of clustering.

![accuracy](accuracy.png?raw=true)
![ami](adjusted_mutual_info_score.png?raw=true)
![ari](adjusted_rand_score.png?raw=true)
![homo](homogeneity_score.png?raw=true)
![nmi](normalized_mutual_info_score.png?raw=true)
![ran](rand_score.png?raw=true)

The confusion matrices for each client are printed

| client 0 | client 1 | client 2 | client 3 |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|![client_0](conf_matrix_c0.png?raw=true)|![client_1](conf_matrix_c1.png?raw=true)|![client_2](conf_matrix_c2.png?raw=true)|![client_3](conf_matrix_c3.png?raw=true)|

| client 4 | client 5 | client 6 | client 7 |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|![client_4](conf_matrix_c4.png?raw=true)|![client_5](conf_matrix_c5.png?raw=true)|![client_6](conf_matrix_c6.png?raw=true)|![client_7](conf_matrix_c7.png?raw=true)|
