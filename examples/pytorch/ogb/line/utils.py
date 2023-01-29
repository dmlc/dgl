import torch


def check_args(args):
    flag = sum([args.only_1st, args.only_2nd])
    assert (
        flag <= 1
    ), "no more than one selection from --only_1st and --only_2nd"
    if flag == 0:
        assert args.dim % 2 == 0, "embedding dimension must be an even number"
    if args.async_update:
        assert args.mix, "please use --async_update with --mix"


def sum_up_params(model):
    """Count the model parameters"""
    n = []
    if model.fst:
        p = model.fst_u_embeddings.weight.cpu().data.numel()
        n.append(p)
        p = model.fst_state_sum_u.cpu().data.numel()
        n.append(p)
    if model.snd:
        p = model.snd_u_embeddings.weight.cpu().data.numel() * 2
        n.append(p)
        p = model.snd_state_sum_u.cpu().data.numel() * 2
        n.append(p)
    n.append(model.lookup_table.cpu().numel())
    try:
        n.append(model.index_emb_negu.cpu().numel() * 2)
    except:
        pass
    print("#params " + str(sum(n)))
