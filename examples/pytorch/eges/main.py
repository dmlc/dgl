import dgl
import torch as th
import torch.optim as optim
import utils
from model import EGES
from sampler import Sampler
from sklearn import metrics
from torch.utils.data import DataLoader


def train(args, train_g, sku_info, num_skus, num_brands, num_shops, num_cates):
    sampler = Sampler(
        train_g,
        args.walk_length,
        args.num_walks,
        args.window_size,
        args.num_negative,
    )
    # for each node in the graph, we sample pos and neg
    # pairs for it, and feed these sampled pairs into the model.
    # (nodes in the graph are of course batched before sampling)
    dataloader = DataLoader(
        th.arange(train_g.num_nodes()),
        # this is the batch_size of input nodes
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: sampler.sample(x, sku_info),
    )
    model = EGES(args.dim, num_skus, num_brands, num_shops, num_cates)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_total_loss = 0
        for step, (srcs, dsts, labels) in enumerate(dataloader):
            # the batch size of output pairs is unfixed
            # TODO: shuffle the triples?
            srcs_embeds, dsts_embeds = model(srcs, dsts)
            loss = model.loss(srcs_embeds, dsts_embeds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()

            if step % args.log_every == 0:
                print(
                    "Epoch {:05d} | Step {:05d} | Step Loss {:.4f} | Epoch Avg Loss: {:.4f}".format(
                        epoch, step, loss.item(), epoch_total_loss / (step + 1)
                    )
                )

        eval(model, test_g, sku_info)

    return model


def eval(model, test_graph, sku_info):
    preds, labels = [], []
    for edge in test_graph:
        src = th.tensor(sku_info[edge.src.numpy()[0]]).view(1, 4)
        dst = th.tensor(sku_info[edge.dst.numpy()[0]]).view(1, 4)
        # (1, dim)
        src = model.query_node_embed(src)
        dst = model.query_node_embed(dst)
        # (1, dim) -> (1, dim) -> (1, )
        logit = th.sigmoid(th.sum(src * dst))
        preds.append(logit.detach().numpy().tolist())
        labels.append(edge.label)

    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)

    print("Evaluate link prediction AUC: {:.4f}".format(metrics.auc(fpr, tpr)))


if __name__ == "__main__":
    args = utils.init_args()

    valid_sku_raw_ids = utils.get_valid_sku_set(args.item_info_data)

    g, sku_encoder, sku_decoder = utils.construct_graph(
        args.action_data, args.session_interval_sec, valid_sku_raw_ids
    )

    train_g, test_g = utils.split_train_test_graph(g)

    sku_info_encoder, sku_info_decoder, sku_info = utils.encode_sku_fields(
        args.item_info_data, sku_encoder, sku_decoder
    )

    num_skus = len(sku_encoder)
    num_brands = len(sku_info_encoder["brand"])
    num_shops = len(sku_info_encoder["shop"])
    num_cates = len(sku_info_encoder["cate"])

    print(
        "Num skus: {}, num brands: {}, num shops: {}, num cates: {}".format(
            num_skus, num_brands, num_shops, num_cates
        )
    )

    model = train(
        args, train_g, sku_info, num_skus, num_brands, num_shops, num_cates
    )
