"""
This code was modified from implementations of SGC in other backends.

Simplifying Graph Convolutional Networks (Wu, Zhang and Souza et al, 2019)
Paper: https://arxiv.org/abs/1902.07153
Author Implementation: https://github.com/Tiiiger/SGC

SGC implementation in DGL.
"""
import argparse
import textwrap

import tensorflow as tf
import tensorflow_addons as tfa

from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.nn.tensorflow.conv import SGConv

_DATASETS = {
    "citeseer": CiteseerGraphDataset(verbose=False),
    "cora": CoraGraphDataset(verbose=False),
    "pubmed": PubmedGraphDataset(verbose=False),
}


def load_data(dataset):
    return _DATASETS[dataset]


def _sum_boolean_tensor(x):
    return tf.reduce_sum(tf.cast(x, dtype="int64"))


def describe_data(data):
    g = data[0]

    n_edges = g.number_of_edges()
    num_classes = data.num_classes

    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    description = textwrap.dedent(
        f"""
        ----Data statistics----
        Edges           {n_edges:,.0f}
        Classes         {num_classes:,.0f}
        Train samples   {_sum_boolean_tensor(train_mask):,.0f}
        Val samples     {_sum_boolean_tensor(val_mask):,.0f}
        Test samples    {_sum_boolean_tensor(test_mask):,.0f}
        """
    )
    return description


class SGC(tf.keras.Model):
    def __init__(self, g, num_classes, bias=False):
        super().__init__()
        self.num_classes = num_classes
        self.g = self.ensure_self_loop(g)
        self.conv = SGConv(
            in_feats=self.in_feats,
            out_feats=self.num_classes,
            k=2,
            cached=True,
            bias=bias,
        )

    def call(self, inputs):
        return self.conv(self.g, inputs)

    @property
    def in_feats(self):
        return self.g.ndata["feat"].shape[1]

    @property
    def num_nodes(self):
        return self.g.num_nodes()

    @staticmethod
    def ensure_self_loop(g):
        g = g.remove_self_loop()
        g = g.add_self_loop()
        return g

    def train_step(self, data):
        X, y = data
        mask = self.g.ndata["train_mask"]

        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = self.compiled_loss(y[mask], y_pred[mask])

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.compiled_metrics.update_state(y[mask], y_pred[mask])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X, y = data
        mask = self.g.ndata["val_mask"]
        y_pred = self(X, training=False)
        self.compiled_loss(y[mask], y_pred[mask])
        self.compiled_metrics.update_state(y[mask], y_pred[mask])
        return {m.name: m.result() for m in self.metrics}

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs, run_eagerly=True)

    def fit(self, *args, **kwargs):
        kwargs["batch_size"] = self.num_nodes
        kwargs["shuffle"] = False
        super().fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        kwargs["batch_size"] = self.num_nodes
        return super().predict(*args, **kwargs)


def main(dataset, lr, bias, n_epochs, weight_decay):
    data = load_data(dataset)
    print(describe_data(data))

    g = data[0]
    X = g.ndata["feat"]
    y = g.ndata["label"]

    model = SGC(g=g, num_classes=data.num_classes, bias=bias)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tfa.optimizers.AdamW(weight_decay, lr)
    accuracy = tf.metrics.SparseCategoricalAccuracy(name="accuracy")

    model.compile(optimizer, loss, metrics=[accuracy])
    model.fit(x=X, y=y, epochs=n_epochs, validation_data=(X, y))

    y_pred = model.predict(X, batch_size=len(X))
    test_mask = g.ndata["test_mask"]
    test_accuracy = accuracy(y[test_mask], y_pred[test_mask])
    print(f"Test Accuracy: {test_accuracy:.1%}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiment for Simple Graph Convolution (SGC)"
    )
    parser.add_argument("--dataset", default="cora", help="dataset to run")
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument(
        "--bias", action="store_true", default=False, help="flag to use bias"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-6, help="weight for L2 loss"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        dataset=args.dataset,
        lr=args.lr,
        bias=args.bias,
        n_epochs=args.n_epochs,
        weight_decay=args.weight_decay,
    )
