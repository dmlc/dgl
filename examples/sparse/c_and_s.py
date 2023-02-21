"""
[Combining Label Propagation and Simple Models Out-performs
Graph Neural Networks](https://arxiv.org/abs/2010.13993)
"""
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from torch.optim import Adam


###############################################################################
# (HIGHLIGHT) Compute Label Propagation with Sparse Matrix API
###############################################################################
@torch.no_grad()
def label_propagation(A_hat, label, num_layers=20, alpha=0.9):
    Y = label
    for _ in range(num_layers):
        Y = alpha * A_hat @ Y + (1 - alpha) * label
        Y = Y.clamp_(0.0, 1.0)
    return Y


def correct(A_hat, label, soft_label, mask):
    # Compute error.
    error = torch.zeros_like(soft_label)
    error[mask] = label[mask] - soft_label[mask]

    # Smooth error.
    smoothed_error = label_propagation(A_hat, error)

    # Autoscale.
    sigma = error[mask].abs()
    sigma = sigma.sum() / sigma.shape[0]
    scale = sigma / smoothed_error.abs().sum(dim=1, keepdim=True)
    scale[scale.isinf() | (scale > 1000)] = 1.0

    # Correct.
    result = soft_label + scale * smoothed_error
    return result


def smooth(A_hat, label, soft_label, mask):
    soft_label[mask] = label[mask].float()
    return label_propagation(A_hat, soft_label)


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(base_model, g, X):
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]

    optimizer = Adam(base_model.parameters(), lr=0.01)

    for epoch in range(10):
        # Forward.
        base_model.train()
        logits = base_model(X)

        # Compute loss with nodes in training set.
        loss = F.cross_entropy(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        base_model.eval()
        logits = base_model(X)
        pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        print(
            f"Base model, In epoch {epoch}, loss: {loss:.3f}, "
            f"val acc: {val_acc:.3f}, test acc: {test_acc:.3f}"
        )
    return logits


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load graph from the existing dataset.
    dataset = CoraGraphDataset()
    g = dataset[0].to(dev)

    # Create the sparse adjacency matrix A.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Calculate the symmetrically normalized adjacency matrix.
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I
    D_hat = dglsp.diag(A_hat.sum(dim=1)) ** -0.5
    A_hat = D_hat @ A_hat @ D_hat

    # Create models.
    X = g.ndata["feat"]
    in_size = X.shape[1]
    out_size = dataset.num_classes
    base_model = nn.Linear(in_size, out_size).to(dev)

    # Stage1: Train the base model.
    logits = train(base_model, g, X)

    # Stage2: Correct and Smooth.
    soft_label = F.softmax(logits, dim=1)
    label = F.one_hot(g.ndata["label"])
    soft_label = correct(A_hat, label, soft_label, g.ndata["train_mask"])
    soft_label = smooth(A_hat, label, soft_label, g.ndata["train_mask"])
    pred = soft_label.argmax(dim=1)
    val_acc, test_acc = evaluate(g, pred)
    print(f"val acc: {val_acc:.3f}, test acc: {test_acc:.3f}")
