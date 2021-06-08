import torch, dgl

def train(model, optimizer, train_data_loader, epochs, deterministic = True,
          KL_weight = 0, reconst_weight = 1, clip_grad_to_norm = None,
          log_interval = None, checkpoint_path = None, checkpoint_interval = None,
          val_data_loader = None, val_interval = None, val_log_interval = None):
    """
    Train the autoencoder to transform a set of input hgraphs
    into a set of corresponding output hgraphs, namely into themselves.

    Args:
    model -
    optimizer -
    train_data_loader -
    epochs -
    deterministic - If the AE is variational, whether latent will be mean or random sample.
    KL_weight - Weight of the KL loss in the net loss.
    reconst_weight - Weight of the reconstruction loss in the net loss.
    clip_grad_to_norm - If not None, gradients will be clipped to this norm.
    log_interval - Training loss will be logged every log_interval epochs.
    val_data_loader -
    val_interval - Validation loss will be recorded every val_interval epochs.
    val_log_interval - Validation loss will be logged every val_log_interval epochs.
    """

    losses = {
        "train": {
            "net": [],
            "reconst": [],
            "KL": []
        },
        "val": {
            "net": [],
            "reconst": [],
            "KL": []
        }
    }

    for epoch in range(epochs):
        epoch_net_loss = 0.0
        epoch_reconst_loss = 0.0
        epoch_KL_loss = 0.0
        for i, batch in enumerate(train_data_loader):
            output_graph, reconst_loss, KL_loss = model(
                batch, target_graphs = batch, deterministic = deterministic
            )
            net_loss = reconst_weight*reconst_loss + KL_weight*KL_loss

            net_loss.backward()
            if clip_grad_to_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_to_norm)
            
            optimizer.step()
            optimizer.zero_grad()

            epoch_net_loss += net_loss.detach().item()
            epoch_reconst_loss += reconst_loss.detach().item()
            epoch_KL_loss += KL_loss.detach().item()
        epoch_net_loss /= i + 1
        epoch_reconst_loss /= i + 1
        epoch_KL_loss /= i + 1

        losses["train"]["net"].append(epoch_net_loss)
        losses["train"]["reconst"].append(epoch_reconst_loss)
        losses["train"]["KL"].append(epoch_KL_loss)

        if checkpoint_interval is not None and epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), checkpoint_path)

        if log_interval is not None and epoch % log_interval == 0:
            print(
                f"Epoch {epoch} | "
                f"Net Loss {epoch_net_loss} | "
                f"Reconst Loss {epoch_reconst_loss} | "
                f"KL Loss {epoch_KL_loss}"
            )
        
        if val_interval is not None and epoch % val_interval == 0:
            val_net_loss, val_reconst_loss, val_KL_loss = validate(
                model, val_data_loader, deterministic, KL_weight, reconst_weight
            )
            
            losses["val"]["net"].append(val_net_loss)
            losses["val"]["reconst"].append(val_reconst_loss)
            losses["val"]["KL"].append(val_KL_loss)

            model.train()
            
            if val_log_interval is not None and epoch % val_log_interval == 0:
                print(
                    f"Validation | "
                    f"Net Loss {val_net_loss} | "
                    f"Reconst Loss {val_reconst_loss} | "
                    f"KL Loss {val_KL_loss}"
                )

def validate(model, val_data_loader,
             deterministic, KL_weight, reconst_weight):
    model.eval()
    val_net_loss = 0.0
    val_reconst_loss = 0.0
    val_KL_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_data_loader):
            output_graph, reconst_loss, KL_loss = model(
                batch, target_graphs = batch, deterministic = deterministic
            )
            net_loss = reconst_weight*reconst_loss + KL_weight*KL_loss
            val_net_loss += net_loss.detach().item()
            val_reconst_loss += reconst_loss.detach().item()
            val_KL_loss += KL_loss.detach().item()
        val_net_loss /= i + 1
        val_reconst_loss /= i + 1
        val_KL_loss /= i + 1

    return val_net_loss, val_reconst_loss, val_KL_loss 
