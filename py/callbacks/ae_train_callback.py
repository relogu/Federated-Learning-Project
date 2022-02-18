import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

def ae_training_callback(
    writer,
    name,
    epoch,
    lr,
    loss,
    dataset,
    config,
    device,
    autoencoder,
):
                    
    val_loss = 0.0
    criterion = MSELoss()
    dataloader = DataLoader(dataset, batch_size=config['ae_batch_size'], shuffle=True)
    for i, val_batch in enumerate(dataloader):
        with torch.no_grad():
            if (
                isinstance(val_batch, tuple) or isinstance(val_batch, list)
            ) and len(val_batch) in [1, 2]:
                val_batch = val_batch[0]
            val_batch = val_batch.to(device)
            validation_output = autoencoder(val_batch)
            loss = criterion(validation_output, val_batch)
            val_loss += loss.cpu().numpy()

    val_loss = (val_loss / (i+1))
    writer.add_scalars(
        "data/autoencoder_{}".format(name),
        {"lr": lr, "loss": loss, "validation_loss": val_loss,},
        epoch,
    )
    return val_loss
