# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines

###################### LIBRARIES #################################################
import warnings
warnings.filterwarnings("ignore")

import torch, faiss
import numpy as np
from scipy import sparse


"""================================================================================================="""
############ LOSS SELECTION FUNCTION #####################
def loss_select(loss, opt, to_optim):
    """
    Selection function which returns the respective criterion while appending to list of trainable parameters if required.

    Args:
        loss:     str, name of loss function to return.
        opt:      argparse.Namespace, contains all training-specific parameters.
        to_optim: list of trainable parameters. Is extend if loss function contains those as well.
    Returns:
        criterion (torch.nn.Module inherited), to_optim (optionally appended)
    """
    if loss == 'smoothap':
        loss_params  = {'anneal':opt.sigmoid_temperature, 'batch_size':opt.bs, "num_id":int(opt.bs / opt.samples_per_class), 'feat_dims':opt.embed_dim}
        criterion    = SmoothAP(**loss_params)
    elif loss == 'ce':
        criterion = CategoricalCrossEntropyLoss()
    elif loss == 'ams':
        criterion = AMSLoss(m=0.3, s=30, num_cls=opt.class_num)
    elif loss == 'softams':
        criterion = AMSLossLogits(m=0.3, s=30, num_cls=opt.class_num)
    else:
        raise Exception('Loss {} not available!'.format(loss))

    return criterion, to_optim


"""==============================================Smooth-AP========================================"""

def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())


class BinarizedF(torch.autograd.Function):
    def forward(self, inp):
        self.save_for_backward(inp)
        a = torch.ones_like(inp)
        b = torch.zeros_like(inp)
        output = torch.where(inp > 0, a, b)
        return output

    def backward(self, output_grad):
        inp, = self.saved_tensors
        input_abs = torch.abs(inp)
        ones = torch.ones_like(inp)
        zeros = torch.zeros_like(inp)
        input_grad = torch.where(input_abs > 0, ones, zeros)
        return input_grad


class BinarizedModule(torch.nn.Module):
    def __init__(self):
        super(BinarizedModule, self).__init__()
        self.BF = BinarizedF()

    def forward(self, inp):
        output = self.BF(inp)
        return output


class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SmoothAP, self).__init__()

        assert(batch_size%num_id==0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """


        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size)
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(preds)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.cuda()
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)
        # compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask.cuda()
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)
        return (1 - ap)


class SoftSmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.

    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.

    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:

        labels = ( A, A, A, B, B, B, C, C, C)

    (the order of the classes however does not matter)

    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.

    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings

    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar

    Examples::

        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, class_num, feat_dims):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SoftSmoothAP, self).__init__()

        self.anneal = anneal
        self.batch_size = batch_size
        self.class_num = class_num
        self.feat_dims = feat_dims

    def sparse_dense_mul(self, s, d):
        ind = s._indices()
        v = s._values()
        dv = d[ind[0, :], ind[1, :], ind[2, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(ind, v * dv, s.size())

    def sparse_dense_mul_(self, s, d):
        ind = s._indices()
        v = s._values()
        dv = d[ind[0, :], ind[2, :], ind[3, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(ind, v * dv, s.size())

    def addc(self, s, c):
        ind = s._indices()
        v = s._values()
        return torch.sparse.FloatTensor(ind, v+1, s.size())

    def forward(self, preds, softlabels):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size)
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(preds)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.cuda()
        # add soft label weights
        # compute the rankings
        sim_sg_sum = torch.sum(sim_sg, dim=-1).unsqueeze(dim=1).repeat(1, self.class_num, 1)
        sim_all_rk = sim_sg_sum + 1
        # sim_all_rk = sim_all_rk * softlabels[:, :, None].cuda()

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        softlabels_t = softlabels.t().reshape(self.class_num, self.batch_size, 1)
        softlabels_t_ = softlabels.t().reshape(self.class_num, 1, self.batch_size)
        softlabel_mask = softlabels_t * softlabels_t_ # c x m x m
        softlabel_mask = torch.where(softlabel_mask > 0, 1.0, 0.0)
        indices = np.where(softlabel_mask != 0)
        values = softlabel_mask[indices]
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = softlabel_mask.shape
        softlabel_mask_sparse = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        pos_mask = 1.0 - torch.eye(self.batch_size)
        pos_mask = pos_mask.unsqueeze(0).repeat(self.class_num, 1, 1)
        pos_mask = self.sparse_dense_mul(softlabel_mask_sparse, pos_mask)
        pos_mask = torch.stack([pos_mask for _ in range(self.batch_size)], dim=0)
        # compute the relevance scores
        sim_pos = compute_aff(preds)
        sim_pos_repeat = sim_pos.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_pos_sg = self.sparse_dense_mul_(pos_mask.cuda(), sigmoid(sim_pos_diff, temp=self.anneal))
        # sim_pos_sg = torch.sum(sim_pos_sg, dim=1)
        # compute the rankings of the positive set
        sim_pos_rk = self.addc(torch.sparse.sum(sim_pos_sg, dim=-1), 1)
        # sim_pos_rk = sim_pos_rk * softlabels[:, :, None].cuda()

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda().requires_grad_(True)
        inds = np.array(sim_pos_rk._indices().cpu().detach())
        classes = np.unique(inds[1])
        groups = np.unique(inds[0])
        for cls in classes:
            for i in groups:
                if softlabels[i][cls] != 0:
                    idx = np.where(softlabel_mask[cls][i] != 0)[0]
                    idx2 = np.where((inds[0] == i) & (inds[1] == cls))
                    if len(idx) >= 4:
                        pos_divide = torch.sum(sim_pos_rk._values()[idx2] / sim_all_rk[i, cls, idx])
                        ap = ap + ((pos_divide) / len(idx) / self.batch_size) * softlabels[i][cls]
        return (1 - ap)

class CategoricalCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, pred_label, target_label):
        return torch.mean(torch.sum(-target_label.cuda() * self.logsoftmax(pred_label), dim=1))


class AMSLoss(torch.nn.Module):
    def __init__(self, m, s, num_cls):
        super(AMSLoss, self).__init__()
        self.m = m
        self.s = s
        self.num_cls = num_cls
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        labels = labels.cuda()
        mask = torch.nn.functional.one_hot(labels.long(), self.num_cls) * self.m
        new_logits = self.s * (logits - mask)
        return self.ce_loss(new_logits, labels.long())


class AMSLossLogits(torch.nn.Module):
    def __init__(self, m, s, num_cls):
        super(AMSLossLogits, self).__init__()
        self.m = m
        self.s = s
        self.num_cls = num_cls
        self.ce_loss = CategoricalCrossEntropyLoss()

    def forward(self, logits, labels):
        labels = labels.cuda()
        mask = labels * self.m
        new_logits = self.s * (logits - mask)
        return self.ce_loss(new_logits, labels)