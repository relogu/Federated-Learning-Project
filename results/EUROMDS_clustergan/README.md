# ClusterGAN

In this simulation the unsupervised deep embedding for clustering is used on the EUROMDS dataset.
The following graphs represents the metrics used to evaluate the performance of clustering.

![accuracy](accuracy.png?raw=true)
![ami](adjusted_mutual_info_score.png?raw=true)
![ari](adjusted_rand_score.png?raw=true)
![homo](homogeneity_score.png?raw=true)
![nmi](normalized_mutual_info_score.png?raw=true)
![ran](rand_score.png?raw=true)

The following graphs represents the losses of the ClusterGAN.

![lat_mse_loss](lat_mse_loss.png?raw=true)
![lat_xe_loss](lat_xe_loss.png?raw=true)
![img_xe_loss](img_mse_loss.png?raw=true)

The confusion matrices for each client are printed

![conf_mat](conf_matrix_nofed.png?raw=true)

Lifelines using Kaplan Meier Fitter are produced

![lifeline](lifelines_pred.png?raw=true)
