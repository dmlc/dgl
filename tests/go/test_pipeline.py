import os

import pytest


@pytest.mark.parametrize(
    "data",
    [
        "cora",
        "citeseer",
        "pubmed",
        "csv",
        "reddit",
        "co-buy-computer",
        "ogbn-arxiv",
        "ogbn-products",
    ],
)
def test_nodepred_data(data):
    os.system(f"dgl configure nodepred --data {data} --model gcn")
    assert os.path.exists(f"nodepred_{data}_gcn.yaml")

    custom_cfg = f"custom_{data}_gcn.yaml"
    os.system(
        f"dgl configure nodepred --data {data} --model gcn --cfg {custom_cfg}"
    )
    assert os.path.exists(custom_cfg)

    custom_script = f"{data}_gcn.py"
    os.system(f"dgl export --cfg {custom_cfg} --output {custom_script}")
    assert os.path.exists(custom_script)


@pytest.mark.parametrize("model", ["gcn", "gat", "sage", "sgc", "gin"])
def test_nodepred_model(model):
    os.system(f"dgl configure nodepred --data cora --model {model}")
    assert os.path.exists(f"nodepred_cora_{model}.yaml")

    custom_cfg = f"custom_cora_{model}.yaml"
    os.system(
        f"dgl configure nodepred --data cora --model {model} --cfg {custom_cfg}"
    )
    assert os.path.exists(custom_cfg)

    custom_script = f"cora_{model}.py"
    os.system(f"dgl export --cfg {custom_cfg} --output {custom_script}")
    assert os.path.exists(custom_script)


@pytest.mark.parametrize(
    "data",
    [
        "cora",
        "citeseer",
        "pubmed",
        "csv",
        "reddit",
        "co-buy-computer",
        "ogbn-arxiv",
        "ogbn-products",
    ],
)
def test_nodepred_ns_data(data):
    os.system(f"dgl configure nodepred-ns --data {data} --model gcn")
    assert os.path.exists(f"nodepred-ns_{data}_gcn.yaml")

    custom_cfg = f"ns-custom_{data}_gcn.yaml"
    os.system(
        f"dgl configure nodepred-ns --data {data} --model gcn --cfg {custom_cfg}"
    )
    assert os.path.exists(custom_cfg)

    custom_script = f"ns-{data}_gcn.py"
    os.system(f"dgl export --cfg {custom_cfg} --output {custom_script}")
    assert os.path.exists(custom_script)


@pytest.mark.parametrize("model", ["gcn", "gat", "sage"])
def test_nodepred_ns_model(model):
    os.system(f"dgl configure nodepred-ns --data cora --model {model}")
    assert os.path.exists(f"nodepred-ns_cora_{model}.yaml")

    custom_cfg = f"ns-custom_cora_{model}.yaml"
    os.system(
        f"dgl configure nodepred-ns --data cora --model {model} --cfg {custom_cfg}"
    )
    assert os.path.exists(custom_cfg)

    custom_script = f"ns-cora_{model}.py"
    os.system(f"dgl export --cfg {custom_cfg} --output {custom_script}")
    assert os.path.exists(custom_script)


@pytest.mark.parametrize(
    "data",
    [
        "cora",
        "citeseer",
        "pubmed",
        "csv",
        "reddit",
        "co-buy-computer",
        "ogbn-arxiv",
        "ogbn-products",
        "ogbl-collab",
        "ogbl-citation2",
    ],
)
def test_linkpred_data(data):
    node_model = "gcn"
    edge_model = "ele"
    neg_sampler = "global"
    custom_cfg = "_".join([data, node_model, edge_model, neg_sampler]) + ".yaml"
    os.system(
        "dgl configure linkpred --data {} --node-model {} --edge-model {} --neg-sampler {} --cfg {}".format(
            data, node_model, edge_model, neg_sampler, custom_cfg
        )
    )
    assert os.path.exists(custom_cfg)

    custom_script = (
        "_".join([data, node_model, edge_model, neg_sampler]) + ".py"
    )
    os.system(
        "dgl export --cfg {} --output {}".format(custom_cfg, custom_script)
    )
    assert os.path.exists(custom_script)


@pytest.mark.parametrize("node_model", ["gcn", "gat", "sage", "sgc", "gin"])
def test_linkpred_node_model(node_model):
    data = "cora"
    edge_model = "ele"
    neg_sampler = "global"
    custom_cfg = "_".join([data, node_model, edge_model, neg_sampler]) + ".yaml"
    os.system(
        "dgl configure linkpred --data {} --node-model {} --edge-model {} --neg-sampler {} --cfg {}".format(
            data, node_model, edge_model, neg_sampler, custom_cfg
        )
    )
    assert os.path.exists(custom_cfg)

    custom_script = (
        "_".join([data, node_model, edge_model, neg_sampler]) + ".py"
    )
    os.system(
        "dgl export --cfg {} --output {}".format(custom_cfg, custom_script)
    )
    assert os.path.exists(custom_script)


@pytest.mark.parametrize("edge_model", ["ele", "bilinear"])
def test_linkpred_edge_model(edge_model):
    data = "cora"
    node_model = "gcn"
    neg_sampler = "global"
    custom_cfg = "_".join([data, node_model, edge_model, neg_sampler]) + ".yaml"
    os.system(
        "dgl configure linkpred --data {} --node-model {} --edge-model {} --neg-sampler {} --cfg {}".format(
            data, node_model, edge_model, neg_sampler, custom_cfg
        )
    )
    assert os.path.exists(custom_cfg)

    custom_script = (
        "_".join([data, node_model, edge_model, neg_sampler]) + ".py"
    )
    os.system(
        "dgl export --cfg {} --output {}".format(custom_cfg, custom_script)
    )
    assert os.path.exists(custom_script)


@pytest.mark.parametrize("neg_sampler", ["global", "persource", ""])
def test_linkpred_neg_sampler(neg_sampler):
    data = "cora"
    node_model = "gcn"
    edge_model = "ele"
    custom_cfg = f"{data}_{node_model}_{edge_model}_{neg_sampler}.yaml"
    if neg_sampler == "":
        os.system(
            "dgl configure linkpred --data {} --node-model {} --edge-model {} --cfg {}".format(
                data, node_model, edge_model, custom_cfg
            )
        )
    else:
        os.system(
            "dgl configure linkpred --data {} --node-model {} --edge-model {} --neg-sampler {} --cfg {}".format(
                data, node_model, edge_model, neg_sampler, custom_cfg
            )
        )
    assert os.path.exists(custom_cfg)

    custom_script = f"{data}_{node_model}_{edge_model}_{neg_sampler}.py"
    os.system(
        "dgl export --cfg {} --output {}".format(custom_cfg, custom_script)
    )
    assert os.path.exists(custom_script)


@pytest.mark.parametrize("data", ["csv", "ogbg-molhiv", "ogbg-molpcba"])
@pytest.mark.parametrize("model", ["gin", "pna"])
def test_graphpred(data, model):
    os.system(
        "dgl configure graphpred --data {} --model {}".format(data, model)
    )
    assert os.path.exists("graphpred_{}_{}.yaml".format(data, model))

    custom_cfg = "custom_{}_{}.yaml".format(data, model)
    os.system(
        "dgl configure graphpred --data {} --model {} --cfg {}".format(
            data, model, custom_cfg
        )
    )
    assert os.path.exists(custom_cfg)

    custom_script = "_".join([data, model]) + ".py"
    os.system(
        "dgl export --cfg {} --output {}".format(custom_cfg, custom_script)
    )
    assert os.path.exists(custom_script)


@pytest.mark.parametrize(
    "recipe",
    [
        "graphpred_hiv_gin.yaml",
        "graphpred_hiv_pna.yaml",
        "graphpred_pcba_gin.yaml",
        "linkpred_cora_sage.yaml",
        "linkpred_citation2_sage.yaml",
        "linkpred_collab_sage.yaml",
        "nodepred_citeseer_gat.yaml",
        "nodepred_citeseer_gcn.yaml",
        "nodepred_citeseer_sage.yaml",
        "nodepred_cora_gat.yaml",
        "nodepred_cora_gcn.yaml",
        "nodepred_cora_sage.yaml",
        "nodepred_pubmed_gat.yaml",
        "nodepred_pubmed_gcn.yaml",
        "nodepred_pubmed_sage.yaml",
        "nodepred-ns_arxiv_gcn.yaml",
        "nodepred-ns_product_sage.yaml",
    ],
)
def test_recipe(recipe):
    # Remove all generated yaml files
    current_dir = os.listdir("./")
    for item in current_dir:
        if item.endswith(".yaml"):
            os.remove(item)

    os.system("dgl recipe get {}".format(recipe))
    assert os.path.exists(recipe)


def test_node_cora():
    os.system("dgl configure nodepred --data cora --model gcn")
    os.system("dgl train --cfg nodepred_cora_gcn.yaml")
    assert os.path.exists("results")
    assert os.path.exists("results/run_0.pth")
    os.system("dgl configure-apply nodepred --cpt results/run_0.pth")
    assert os.path.exists("apply_nodepred_cora_gcn.yaml")
    os.system(
        "dgl configure-apply nodepred --data cora --cpt results/run_0.pth --cfg apply.yaml"
    )
    assert os.path.exists("apply.yaml")
    os.system("dgl apply --cfg apply.yaml")
    assert os.path.exists("apply_results/output.csv")
    os.system("dgl export --cfg apply.yaml --output apply.py")
    assert os.path.exists("apply.py")
