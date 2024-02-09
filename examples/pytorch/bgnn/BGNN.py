import itertools
import time
from collections import defaultdict as ddict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, sum_models
from sklearn import preprocessing
from sklearn.metrics import r2_score
from tqdm import tqdm


class BGNNPredictor:
    """
    Description
    -----------
    Boost GNN predictor for semi-supervised node classification or regression problems.
    Publication: https://arxiv.org/abs/2101.08543

    Parameters
    ----------
    gnn_model : nn.Module
        DGL implementation of GNN model.
    task: str, optional
        Regression or classification task.
    loss_fn : callable, optional
        Function that takes torch tensors, pred and true, and returns a scalar.
    trees_per_epoch : int, optional
        Number of GBDT trees to build each epoch.
    backprop_per_epoch : int, optional
        Number of backpropagation steps to make each epoch.
    lr : float, optional
        Learning rate of gradient descent optimizer.
    append_gbdt_pred : bool, optional
        Append GBDT predictions or replace original input node features.
    train_input_features : bool, optional
        Train original input node features.
    gbdt_depth : int, optional
        Depth of each tree in GBDT model.
    gbdt_lr : float, optional
        Learning rate of GBDT model.
    gbdt_alpha : int, optional
        Weight to combine previous and new GBDT trees.
    random_seed : int, optional
        random seed for GNN and GBDT models.

    Examples
    ----------
    gnn_model = GAT(10, 20, num_heads=5),
    bgnn = BGNNPredictor(gnn_model)
    metrics = bgnn.fit(graph, X, y, train_mask, val_mask, test_mask, cat_features)
    """

    def __init__(
        self,
        gnn_model,
        task="regression",
        loss_fn=None,
        trees_per_epoch=10,
        backprop_per_epoch=10,
        lr=0.01,
        append_gbdt_pred=True,
        train_input_features=False,
        gbdt_depth=6,
        gbdt_lr=0.1,
        gbdt_alpha=1,
        random_seed=0,
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model = gnn_model.to(self.device)
        self.task = task
        self.loss_fn = loss_fn
        self.trees_per_epoch = trees_per_epoch
        self.backprop_per_epoch = backprop_per_epoch
        self.lr = lr
        self.append_gbdt_pred = append_gbdt_pred
        self.train_input_features = train_input_features
        self.gbdt_depth = gbdt_depth
        self.gbdt_lr = gbdt_lr
        self.gbdt_alpha = gbdt_alpha
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def init_gbdt_model(self, num_epochs, epoch):
        if self.task == "regression":
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = "RMSE"
        else:
            if epoch == 0:  # we predict multiclass probs at first epoch
                catboost_model_obj = CatBoostClassifier
                catboost_loss_fn = "MultiClass"
            else:  # we predict the gradients for each class at epochs > 0
                catboost_model_obj = CatBoostRegressor
                catboost_loss_fn = "MultiRMSE"

        return catboost_model_obj(
            iterations=num_epochs,
            depth=self.gbdt_depth,
            learning_rate=self.gbdt_lr,
            loss_function=catboost_loss_fn,
            random_seed=self.random_seed,
            nan_mode="Min",
        )

    def fit_gbdt(self, pool, trees_per_epoch, epoch):
        gbdt_model = self.init_gbdt_model(trees_per_epoch, epoch)
        gbdt_model.fit(pool, verbose=False)
        return gbdt_model

    def append_gbdt_model(self, new_gbdt_model, weights):
        if self.gbdt_model is None:
            return new_gbdt_model
        return sum_models([self.gbdt_model, new_gbdt_model], weights=weights)

    def train_gbdt(
        self,
        gbdt_X_train,
        gbdt_y_train,
        cat_features,
        epoch,
        gbdt_trees_per_epoch,
        gbdt_alpha,
    ):
        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features)
        epoch_gbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch, epoch)
        if epoch == 0 and self.task == "classification":
            self.base_gbdt = epoch_gbdt_model
        else:
            self.gbdt_model = self.append_gbdt_model(
                epoch_gbdt_model, weights=[1, gbdt_alpha]
            )

    def update_node_features(self, node_features, X, original_X):
        # get predictions from gbdt model
        if self.task == "regression":
            predictions = np.expand_dims(
                self.gbdt_model.predict(original_X), axis=1
            )
        else:
            predictions = self.base_gbdt.predict_proba(original_X)
            if self.gbdt_model is not None:
                predictions_after_one = self.gbdt_model.predict(original_X)
                predictions += predictions_after_one

        # update node features with predictions
        if self.append_gbdt_pred:
            if self.train_input_features:
                predictions = np.append(
                    node_features.detach().cpu().data[:, : -self.out_dim],
                    predictions,
                    axis=1,
                )  # replace old predictions with new predictions
            else:
                predictions = np.append(
                    X, predictions, axis=1
                )  # append original features with new predictions

        predictions = torch.from_numpy(predictions).to(self.device)

        node_features.data = predictions.float().data

    def update_gbdt_targets(
        self, node_features, node_features_before, train_mask
    ):
        return (
            (node_features - node_features_before)
            .detach()
            .cpu()
            .numpy()[train_mask, -self.out_dim :]
        )

    def init_node_features(self, X):
        node_features = torch.empty(
            X.shape[0], self.in_dim, requires_grad=True, device=self.device
        )
        if self.append_gbdt_pred:
            node_features.data[:, : -self.out_dim] = torch.from_numpy(
                X.to_numpy(copy=True)
            )
        return node_features

    def init_optimizer(
        self, node_features, optimize_node_features, learning_rate
    ):
        params = [self.model.parameters()]
        if optimize_node_features:
            params.append([node_features])
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=learning_rate)
        return optimizer

    def train_model(self, model_in, target_labels, train_mask, optimizer):
        y = target_labels[train_mask]

        self.model.train()
        logits = self.model(*model_in).squeeze()
        pred = logits[train_mask]

        if self.loss_fn is not None:
            loss = self.loss_fn(pred, y)
        else:
            if self.task == "regression":
                loss = torch.sqrt(F.mse_loss(pred, y))
            elif self.task == "classification":
                loss = F.cross_entropy(pred, y.long())
            else:
                raise NotImplemented(
                    "Unknown task. Supported tasks: classification, regression."
                )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def evaluate_model(self, logits, target_labels, mask):
        metrics = {}
        y = target_labels[mask]
        with torch.no_grad():
            pred = logits[mask]
            if self.task == "regression":
                metrics["loss"] = torch.sqrt(
                    F.mse_loss(pred, y).squeeze() + 1e-8
                )
                metrics["rmsle"] = torch.sqrt(
                    F.mse_loss(torch.log(pred + 1), torch.log(y + 1)).squeeze()
                    + 1e-8
                )
                metrics["mae"] = F.l1_loss(pred, y)
                metrics["r2"] = torch.Tensor(
                    [r2_score(y.cpu().numpy(), pred.cpu().numpy())]
                )
            elif self.task == "classification":
                metrics["loss"] = F.cross_entropy(pred, y.long())
                metrics["accuracy"] = torch.Tensor(
                    [(y == pred.max(1)[1]).sum().item() / y.shape[0]]
                )

            return metrics

    def train_and_evaluate(
        self,
        model_in,
        target_labels,
        train_mask,
        val_mask,
        test_mask,
        optimizer,
        metrics,
        gnn_passes_per_epoch,
    ):
        loss = None

        for _ in range(gnn_passes_per_epoch):
            loss = self.train_model(
                model_in, target_labels, train_mask, optimizer
            )

        self.model.eval()
        logits = self.model(*model_in).squeeze()
        train_results = self.evaluate_model(logits, target_labels, train_mask)
        val_results = self.evaluate_model(logits, target_labels, val_mask)
        test_results = self.evaluate_model(logits, target_labels, test_mask)
        for metric_name in train_results:
            metrics[metric_name].append(
                (
                    train_results[metric_name].detach().item(),
                    val_results[metric_name].detach().item(),
                    test_results[metric_name].detach().item(),
                )
            )
        return loss

    def update_early_stopping(
        self,
        metrics,
        epoch,
        best_metric,
        best_val_epoch,
        epochs_since_last_best_metric,
        metric_name,
        lower_better=False,
    ):
        train_metric, val_metric, test_metric = metrics[metric_name][-1]
        if (lower_better and val_metric < best_metric[1]) or (
            not lower_better and val_metric > best_metric[1]
        ):
            best_metric = metrics[metric_name][-1]
            best_val_epoch = epoch
            epochs_since_last_best_metric = 0
        else:
            epochs_since_last_best_metric += 1
        return best_metric, best_val_epoch, epochs_since_last_best_metric

    def log_epoch(
        self,
        pbar,
        metrics,
        epoch,
        loss,
        epoch_time,
        logging_epochs,
        metric_name="loss",
    ):
        train_metric, val_metric, test_metric = metrics[metric_name][-1]
        if epoch and epoch % logging_epochs == 0:
            pbar.set_description(
                "Epoch {:05d} | Loss {:.3f} | Loss {:.3f}/{:.3f}/{:.3f} | Time {:.4f}".format(
                    epoch,
                    loss,
                    train_metric,
                    val_metric,
                    test_metric,
                    epoch_time,
                )
            )

    def fit(
        self,
        graph,
        X,
        y,
        train_mask,
        val_mask,
        test_mask,
        original_X=None,
        cat_features=None,
        num_epochs=100,
        patience=10,
        logging_epochs=1,
        metric_name="loss",
    ):
        """

        :param graph : dgl.DGLGraph
            Input graph
        :param X : pd.DataFrame
            Input node features. Each column represents one input feature. Each row is a node.
            Values in dataframe are numerical, after preprocessing.
        :param y : pd.DataFrame
            Input node targets. Each column represents one target. Each row is a node
            (order of nodes should be the same as in X).
        :param train_mask : list[int]
            Node indexes (rows) that belong to train set.
        :param val_mask : list[int]
            Node indexes (rows) that belong to validation set.
        :param test_mask : list[int]
            Node indexes (rows) that belong to test set.
        :param original_X : pd.DataFrame, optional
            Input node features before preprocessing. Each column represents one input feature. Each row is a node.
            Values in dataframe can be of any type, including categorical (e.g. string, bool) or
            missing values (None). This is useful if you want to preprocess X with GBDT model.
        :param cat_features: list[int]
            Feature indexes (columns) which are categorical features.
        :param num_epochs : int
            Number of epochs to run.
        :param patience : int
            Number of epochs to wait until early stopping.
        :param logging_epochs : int
            Log every n epoch.
        :param metric_name : str
            Metric to use for early stopping.
        :param normalize_features : bool
            If to normalize original input features X (column wise).
        :param replace_na: bool
            If to replace missing values (None) in X.
        :return: metrics evaluated during training
        """

        # initialize for early stopping and metrics
        if metric_name in ["r2", "accuracy"]:
            best_metric = [np.cfloat("-inf")] * 3  # for train/val/test
        else:
            best_metric = [np.cfloat("inf")] * 3  # for train/val/test

        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list)
        if cat_features is None:
            cat_features = []

        if self.task == "regression":
            self.out_dim = y.shape[1]
        elif self.task == "classification":
            self.out_dim = len(set(y.iloc[test_mask, 0]))
        self.in_dim = (
            self.out_dim + X.shape[1] if self.append_gbdt_pred else self.out_dim
        )

        if original_X is None:
            original_X = X.copy()
            cat_features = []

        gbdt_X_train = original_X.iloc[train_mask]
        gbdt_y_train = y.iloc[train_mask]
        gbdt_alpha = self.gbdt_alpha
        self.gbdt_model = None

        node_features = self.init_node_features(X)
        optimizer = self.init_optimizer(
            node_features, optimize_node_features=True, learning_rate=self.lr
        )

        y = (
            torch.from_numpy(y.to_numpy(copy=True))
            .float()
            .squeeze()
            .to(self.device)
        )
        graph = graph.to(self.device)

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()

            # gbdt part
            self.train_gbdt(
                gbdt_X_train,
                gbdt_y_train,
                cat_features,
                epoch,
                self.trees_per_epoch,
                gbdt_alpha,
            )

            self.update_node_features(node_features, X, original_X)
            node_features_before = node_features.clone()
            model_in = (graph, node_features)
            loss = self.train_and_evaluate(
                model_in,
                y,
                train_mask,
                val_mask,
                test_mask,
                optimizer,
                metrics,
                self.backprop_per_epoch,
            )
            gbdt_y_train = self.update_gbdt_targets(
                node_features, node_features_before, train_mask
            )

            self.log_epoch(
                pbar,
                metrics,
                epoch,
                loss,
                time.time() - start2epoch,
                logging_epochs,
                metric_name=metric_name,
            )

            # check early stopping
            (
                best_metric,
                best_val_epoch,
                epochs_since_last_best_metric,
            ) = self.update_early_stopping(
                metrics,
                epoch,
                best_metric,
                best_val_epoch,
                epochs_since_last_best_metric,
                metric_name,
                lower_better=(metric_name not in ["r2", "accuracy"]),
            )
            if patience and epochs_since_last_best_metric > patience:
                break

            if np.isclose(gbdt_y_train.sum(), 0.0):
                print("Node embeddings do not change anymore. Stopping...")
                break

        print(
            "Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}".format(
                metric_name, best_val_epoch, *best_metric
            )
        )
        return metrics

    def predict(self, graph, X, test_mask):
        graph = graph.to(self.device)
        node_features = torch.empty(X.shape[0], self.in_dim).to(self.device)
        self.update_node_features(node_features, X, X)
        logits = self.model(graph, node_features).squeeze()
        if self.task == "regression":
            return logits[test_mask]
        else:
            return logits[test_mask].max(1)[1]

    def plot_interactive(
        self,
        metrics,
        legend,
        title,
        logx=False,
        logy=False,
        metric_name="loss",
        start_from=0,
    ):
        import plotly.graph_objects as go

        metric_results = metrics[metric_name]
        xs = [list(range(len(metric_results)))] * len(metric_results[0])
        ys = list(zip(*metric_results))

        fig = go.Figure()
        for i in range(len(ys)):
            fig.add_trace(
                go.Scatter(
                    x=xs[i][start_from:],
                    y=ys[i][start_from:],
                    mode="lines+markers",
                    name=legend[i],
                )
            )

        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title="Epoch",
            yaxis_title=metric_name,
            font=dict(
                size=40,
            ),
            height=600,
        )

        if logx:
            fig.update_layout(xaxis_type="log")
        if logy:
            fig.update_layout(yaxis_type="log")

        fig.show()
