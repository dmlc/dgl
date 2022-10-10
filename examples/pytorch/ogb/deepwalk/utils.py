import torch


def shuffle_walks(walks):
    seeds = torch.randperm(walks.size()[0])
    return walks[seeds]


def sum_up_params(model):
    """Count the model parameters"""
    n = []
    n.append(model.u_embeddings.weight.cpu().data.numel() * 2)
    n.append(model.lookup_table.cpu().numel())
    n.append(model.index_emb_posu.cpu().numel() * 2)
    n.append(model.grad_u.cpu().numel() * 2)

    try:
        n.append(model.index_emb_negu.cpu().numel() * 2)
    except:
        pass
    try:
        n.append(model.state_sum_u.cpu().numel() * 2)
    except:
        pass
    try:
        n.append(model.grad_avg.cpu().numel())
    except:
        pass
    try:
        n.append(model.context_weight.cpu().numel())
    except:
        pass

    print("#params " + str(sum(n)))
    exit()
