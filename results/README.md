
Simulations Results Collection
==============================



Here are presented the results to some models tested.
The configurations of the models are characterized by a 
letter(['a' 'b' 'c' 'd' 'e' 'f' 'g']) and then compared by the number of clusters.
  
The following table resumes the main differences between configurations

|Letter|AE epochs|Dropout rate|Random Flip rate|Unit Norm reg.|
| :---: | :---: | :---: | :---: | :---: |
|a|2500|0.2|0.2|No|
|b|2500|0.05|0.05|No|
|c|2500|0.2|0.2|Yes|
|d|2500|0.1|0.1|Yes|
|e|2500|0.05|0.05|Yes|
|f|2500|0.01|0.01|Yes|
|g|5000|0.01|0.01|Yes|



Every model is trained in the clustering part for at least 2000 epochs.
After 2000 epochs, the algorithm checks how 
many test elements change their label.
When the number of the element which change their label is zero, the training is 
stopped.
These last two sentences do not hold for the federated models, where the clustering training lasts until the 
last pre-define epoch (10000 epochs in total).
Four different federated set up were tested, with 2, 4, 6, and 8 clients.

In every federated set up the clients are equipotential with an equal number of data samples.

Not Federated DEC
=================

AutoEncoder (AE)
----------------

### Pretraining
  

Final values for G metric, computed as the ratio between training and evaluation losses.  

![pretrain ae G metric](Pretrain_SAE_G_metric.png?raw=true)

### Finetuning
  

Final values for G metric, computed as the ratio between training and evaluation losses.  

![finetune ae G metric](Finetune_SAE_G_metric.png?raw=true)

Clustering Model
----------------
  

Final values for G metric, computed as the ratio between training and evaluation losses.  
![clustering G metric](Clustering_Model_G_metric.png?raw=true)
  

Final values for evaluation cycle accuracy metric.  
![clustering cycle acc metric](clustering_cycle_acc_metric.png?raw=true)

Federated DEC
=============

AutoEncoder (AE)
----------------

### Pretraining
  

Final values for G metric, computed as the ratio between training and evaluation losses.  

![fed pretrain ae G metric](fed_Pretrain_SAE_G_metric.png?raw=true)

### Finetuning
  

Final values for G metric, computed as the ratio between training and evaluation losses.  

![fed finetune ae G metric](fed_Finetune_SAE_G_metric.png?raw=true)

Clustering Model
----------------
  

Final values for G metric, computed as the ratio between training and evaluation losses.  
![fed clustering G metric](fed_Clustering_Model_G_metric.png?raw=true)
  

Final values for evaluation cycle accuracy metric.  
![fed clustering cycle acc metric](fed_clustering_cycle_acc_metric.png?raw=true)

Comparison vs HDP
-----------------



Some objective metrics to measure the information of HDP labels carried by DEC assignments. Here are shown the DEC with
 6 clusters (same number as HDP) and with 8 clusters (the best performing).
  
![vs HDP metrics 6](metrics6.png?raw=true)
  
![vs HDP metrics 8](metrics8.png?raw=true)



Some objective metrics to measure the clustering performance of HDP on data space and the performance of DEC on feature
 space. Here are shown the DEC with 6 clusters (same number as HDP) and with 8 clusters (the best performing). 
Silhouette score, Calinski Harabasz score and Davies Bouldin score are used. Different clients set up are compared 
versus HDP performance  
![silhouette 6](silh_score6.png?raw=true)
  
![silhouette 8](silh_score8.png?raw=true)
  
![calinski harabasz 6](cal_har_score6.png?raw=true)
  
![calinski harabasz 8](cal_har_score8.png?raw=true)
  
![davies bouldin 6](dav_bou_score6.png?raw=true)
  
![davies bouldin 8](dav_bou_score8.png?raw=true)
