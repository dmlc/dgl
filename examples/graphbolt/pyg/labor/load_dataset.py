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
    multilabel = name in ["yelp"]
    return dataset, multilabel


def load_dataset(dataset_name, disk_based_feature_keys=None):
    multilabel = False
    if dataset_name in [
        "reddit",
        "cora",
        "citeseer",
        "pubmed",
        "yelp",
        "flickr",
    ]:
        dataset, multilabel = load_dgl(dataset_name)
    else:
        if "mag240M" in dataset_name:
            dataset_name = "ogb-lsc-mag240m"
        dataset = gb.BuiltinDataset(dataset_name)
        if disk_based_feature_keys is None:
            disk_based_feature_keys = set()
        for feature in dataset.yaml_data["feature_data"]:
            feature_key = (feature["domain"], feature["type"], feature["name"])
            # Set the in_memory setting to False without modifying YAML file.
            if feature_key in disk_based_feature_keys:
                feature["in_memory"] = False
        dataset = dataset.load()

    return dataset, multilabel
