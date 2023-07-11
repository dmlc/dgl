import dgl
import dgl.graphbolt
import pytest
import torch


def get_graphbolt_fetch_func():
    feature_store = {
        "feature": dgl.graphbolt.TorchBasedFeatureStore(torch.randn(200, 4)),
        "label": dgl.graphbolt.TorchBasedFeatureStore(
            torch.randint(0, 10, (200,))
        ),
    }

    def fetch_func(data):
        return feature_store["feature"].read(data), feature_store["label"].read(
            data
        )

    return fetch_func


def get_tensor_fetch_func():
    feature_store = torch.randn(200, 4)
    label = torch.randint(0, 10, (200,))

    def fetch_func(data):
        return feature_store[data], label[data]

    return fetch_func


@pytest.mark.parametrize(
    "fetch_func", [get_graphbolt_fetch_func(), get_tensor_fetch_func()]
)
def test_FeatureFetcher(fetch_func):
    itemset = dgl.graphbolt.ItemSet(torch.arange(10))
    minibatch_dp = dgl.graphbolt.MinibatchSampler(itemset, batch_size=2)
    fetcher_dp = dgl.graphbolt.FeatureFetcher(minibatch_dp, fetch_func)

    assert len(list(fetcher_dp)) == 5
