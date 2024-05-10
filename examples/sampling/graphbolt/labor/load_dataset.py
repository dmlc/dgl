import dgl.graphbolt as gb


def load_dgl(name):
    from dgl.data import (
        CiteseerGraphDataset,
        CoraGraphDataset,
        FlickrDataset,
        PubmedGraphDataset,
        RedditDataset,
        YelpDataset,
    )

    d = {
        "cora": CoraGraphDataset,
        "citeseer": CiteseerGraphDataset,
        "pubmed": PubmedGraphDataset,
        "reddit": RedditDataset,
        "yelp": YelpDataset,
        "flickr": FlickrDataset,
    }

    dataset = gb.LegacyDataset(d[name]())
    new_feature = gb.TorchBasedFeatureStore([])
    new_feature._features = dataset.feature._features
    dataset._feature = new_feature
    return dataset


def load_dataset(dataset_name):
    multilabel = dataset_name in ["yelp"]
    if dataset_name in [
        "reddit",
        "cora",
        "citeseer",
        "pubmed",
        "yelp",
        "flickr",
    ]:
        dataset = load_dgl(dataset_name)
        # if multilabel:
        #     g.ndata["labels"] = g.ndata["labels"].to(dtype=th.float32)
    elif dataset_name in [
        "ogbn-products",
        "ogbn-arxiv",
        "ogbn-papers100M",
        "ogbn-mag240M",
    ]:
        if "mag240M" in dataset_name:
            dataset_name = "ogb-lsc-mag240m"
        dataset = gb.BuiltinDataset(dataset_name).load()
    else:
        raise ValueError("unknown dataset")

    return dataset, multilabel
