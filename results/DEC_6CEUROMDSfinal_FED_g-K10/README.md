
Simulation Results
==================

Learning Curves
===============

AutoEncoder (AE)
----------------
  
Pretraining  

![pretrain ae](pretrain_ae_loss.png?raw=true)
  

![pretrain ae](pretrain_ae_metric.png?raw=true)
  
Finetuning  

![finetune ae](finetune_ae_loss.png?raw=true)
  

![finetune ae](finetune_ae_metric.png?raw=true)

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

Predicted Clusters' Properties
------------------------------

||||||
| :---: | :---: | :---: | :---: | :---: |
|Lossofchr5ordel5qPLUSother|PTPN11|del5q|GATA2|U2AF1|
|IDH2|LossofchrY|Isochr17qort17p|KIT|ZRSR2|
|Lossofchr7ordel7q|DNMT3A|FLT3|NF1|Lossofchr11ordel11q|
|NPM1|TP53|NRAS|Lossofchr9ordel9q|ASXL1|
|PHF6|TET2|idicXq13|BCOR|SF3B1|
|SRSF2|JAK2|Lossofchr20ordel20q|CBL|KRAS|
|WT1|Lossofchr13ordel13q|IDH1|CEBPA|ETV6|
|Gainofchr8|EZH2|STAG2|Lossofchr12ordel12port12p|RUNX1|
  
![centroids](pred_imgs.png?raw=true)

Lifelines
---------
  
![lifelines](lifelines.png?raw=true)

Reduced Representation using t-SNE
----------------------------------
  
![tsne_feat_pred_labels](tsne_feat_pred_labels.png?raw=true)
  
![tsne_data_pred_labels](tsne_data_pred_labels.png?raw=true)
