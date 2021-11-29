#>>> dgl-enter: skip-codegen
from ...data import DataConfigOption
from ...model import ModelConfigOption, model_gen
from ...config import PipelineConfigOption, OptimConfigOption, LossConfigOption

opt = PipelineConfigOption(name='nodepred')
opt.add_option_class('data', DataConfigOption)
opt.add_option_class('model', ModelConfigOption)
opt.add_option_class('loss', LossConfigOption)
opt.add_option_class('optimizer', OptimConfigOption)
opt.add_option('use_node_feat', type=bool, default=True,
               help='whether to use dataset provided node feature')
opt.add_option('use_node_embed', type=bool, default=False,
               help='whether to init a learnable node embedding')
opt.add_option('num_epochs', type=int, default=200,
               help='number of training epochs')
opt.add_option('eval_period', type=20, default=20,
               help='number of training epochs between each evaluation')
opt.add_option('checkpoint_path', type=str, default='checkpoint.pt',
               help='model checkpoint path')
opt.add_option('patience', type=int, default=50,
               help='early stop patience; if -1, no early stopping')

from ...codegen import model_gen, optim_gen
def code_generation(cfg):
    mode_code = model_gen(cfg)
    code = f'{model_code}\n print("hello")\n'
    return code
#<<< dgl-enter: skip-codegen

import torch
import torch.nn.functional as F
import dgl

class EarlyStopping:
    def __init__(self,
                 patience: int = -1,
                 checkpoint_path: str = 'checkpoint.pt'):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Save model when validation loss decreases.'''
        torch.save(model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def train(cfg, data, model):
    g = data[0]  # Only train on the first graph
    g = g.to(device)
    node_feat = g.ndata['feat']
    edge_feat = g.edata.get('feat', None)
    label = g.ndata['label']
    train_mask, val_mask, test_mask = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']

    optim_class = getattr(torch.optim, cfg['optimizer'])
    optimizer = optim_class(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    stopper = EarlyStopping(cfg['patience'], cfg['checkpoint_path'])

    for epoch in range(cfg['num_epochs']):
        model.train()
        logits = model(g, node_feat, edge_feat)
        loss = loss_fcn(logits[train_mask], label[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], label[train_mask])
        if epoch != 0 and epoch % cfg['eval_period'] == 0:
            val_acc = accuracy(logits[val_mask], label[val_mask])
            if stopper.step(val_acc, model):
                break
        print("Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f} | ValAcc {:.4f}".
              format(epoch, loss.item(), train_acc, val_acc))

    stopper.load_checkpoint(model)
    model.eval()
    with torch.no_grad():
        logits = model(g, node_feat, edge_feat)
        test_acc = accuracy(logits[test_mask], labels[test_mask])
    print("Test Accuracy {:.4f}".format(test_acc))
