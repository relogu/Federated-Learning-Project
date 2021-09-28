
Simulation Results
==================

Learning Curves
===============

AutoEncoder (AE)
----------------
  
Pretraining  

![pretrain ae](pretrain_ae_history.png?raw=true)
  
Finetuning  

![finetune ae](finetune_ae_history.png?raw=true)

Clustering Model
----------------
  
![clustering_loss](clustering_loss.png?raw=true)
  
![clustering_G_metric](clustering_G_metric.png?raw=true)
  
![clustering_G_metric](clustering_tol_ca_metrics.png?raw=true)

Model Results
=============

Labels Distribution
-------------------
  
![labels distr](labels_distribution.png?raw=true)

Cycle Accuracy per Cluster
--------------------------
  
![conf matrix](conf_matrix_cycle_accuracy.png?raw=true)

Predicted Clusters' Properties
------------------------------

||||||
| :---: | :---: | :---: | :---: | :---: |
|Lossofchr9ordel9q|EZH2|STAG2|Lossofchr13ordel13q|idicXq13|
|CBL|GATA2|FLT3|TET2|ASXL1|
|Lossofchr20ordel20q|NF1|ETV6|Lossofchr12ordel12port12p|NPM1|
|SF3B1|del5q|DNMT3A|PHF6|SRSF2|
|CEBPA|PTPN11|LossofchrY|WT1|KRAS|
|Gainofchr8|Lossofchr5ordel5qPLUSother|Lossofchr7ordel7q|KIT|IDH1|
|NRAS|U2AF1|Lossofchr11ordel11q|IDH2|RUNX1|
|JAK2|BCOR|Isochr17qort17p|TP53|ZRSR2|
  
Extracted from real data belonging to the cluster
  
![centroids samples](data_pred_imgs.png?raw=true)
  
Extracted from samples near the cluster centroid in the feature space
  
![centroids data](samples_pred_imgs.png?raw=true)

Lifelines
---------
  
![lifelines](lifelines.png?raw=true)

Reduced Representation using t-SNE
----------------------------------


Here is represented the reduced, by t-SNE, representation of the test set. With a gray cross are represented the real 
data sample that mostly represent the cluster from a topological point of view (euclidean norm in feature space)  
![tsne_feat_pred_labels](tsne_feat_pred_labels.png?raw=true)
  
![tsne_data_pred_labels](tsne_data_pred_labels.png?raw=true)
