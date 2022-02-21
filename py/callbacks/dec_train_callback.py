import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from py.dec.torch.utils import cluster_accuracy


def dec_train_callback(
    writer,
    config,
    dataset,
    model,
    autoencoder,
    device,
    epoch,
    predicted_previous,
):

    cl_recon = 0.0
    criterion = MSELoss()
    data = []
    r_data = []
    features = []
    prob_labels = []
    r_prob_labels = []
    actual = []
    model.eval()
    autoencoder.eval()
    dataloader = DataLoader(dataset, batch_size=config['ae_batch_size'], shuffle=False)
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # unpack if we have a prediction label
                actual.append(value.cpu())
                data.append(batch.cpu())
            batch = batch.to(device, non_blocking=True)
            r_batch = autoencoder(batch)
            f_batch = autoencoder.encoder(batch)
            features.append(f_batch.cpu())
            r_data.append(r_batch.cpu())
            loss = criterion(r_batch, batch)
            cl_recon += loss.cpu().numpy()
            prob_labels.append(model(batch).cpu())
            r_prob_labels.append(model(r_batch).cpu())
            
    cl_recon = (cl_recon / (i+1))
    predicted = torch.cat(prob_labels).max(1)[1]
    r_predicted = torch.cat(r_prob_labels).max(1)[1]
    actual = torch.cat(actual).long()
    data = torch.cat(data).numpy()
    r_data = torch.cat(r_data).numpy()
    features = torch.cat(features).numpy()
    delta_label = (
        float((predicted != predicted_previous).float().sum().item())
        / predicted_previous.shape[0]
    )

    cos_sil_score = silhouette_score(
        X=data,
        labels=predicted,
        metric='cosine')

    eucl_sil_score = silhouette_score(
        X=features,
        labels=predicted,
        metric='euclidean')
    
    data_calinski_harabasz = calinski_harabasz_score(
        X=data,
        labels=predicted)

    feat_calinski_harabasz = calinski_harabasz_score(
        X=features,
        labels=predicted)

    reassignment, accuracy = cluster_accuracy(predicted.numpy(), actual.numpy())
    r_reassignment, cycle_accuracy = cluster_accuracy(r_predicted.numpy(), predicted.numpy())
    writer.add_scalars(
        "data/clustering_alpha{}".format(config['alpha']),
        {
            "accuracy": accuracy,
            "cycle_accuracy": cycle_accuracy,
            "loss": loss,
            "delta_label": delta_label,
            "cl_recon": cl_recon,
            'cos_sil_score': cos_sil_score,
            'eucl_sil_score': eucl_sil_score,
            'data_calinski_harabasz': data_calinski_harabasz,
            'feat_calinski_harabasz': feat_calinski_harabasz,
         },
        epoch,
    )
    return predicted
