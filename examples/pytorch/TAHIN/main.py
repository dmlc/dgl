import torch 
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
import pickle as pkl
import argparse
from data_loader import load_data
from model_gat import LinkPredict
from utils.utils import *

def main(args):
    #step 1: Check device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    #step 2: Load data
    g, train_loader, eval_loader, test_loader, meta_paths, user_key, item_key = load_data(args.dataset, args.batch, args.num_workers)
    g = g.to(device)
    print('Data loaded.')

    #step 3: Create model and training components
    '''
    model = LinkPredict(
        g, meta_paths, user_key, item_key, args.in_size, args.hidden_size, args.layer_num_heads, args.layer_dropout, args.out_size, args.dropout
        )
    '''
    model = LinkPredict(
        args.in_size, args.hidden_size, args.layer_num_heads, args.layer_dropout, args.out_size, args.dropout
        )

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print('Model created.')

    
    #step 4: Training
    print('Start training.')
    best_acc = 0.0
    kill_cnt = 0

    for i, ntype in enumerate(g.ntypes):
        if ntype == user_key:
            user_id = i
        if ntype == item_key:
            item_id = i
    initializer = nn.init.xavier_uniform_
    homo_g = dgl.to_homogeneous(g)
    homo_g = dgl.add_self_loop(homo_g)
    feats = initializer(torch.empty(homo_g.num_nodes(), args.in_size)).to(device)
    feats.requires_grad = True
    node_ids = torch.arange(homo_g.num_nodes())
    node_tids = homo_g.ndata[dgl.NTYPE]
    user_loc = (node_tids == user_id)
    item_loc = (node_tids == item_id)
    user_target_idx = node_ids[user_loc]
    item_target_idx = node_ids[item_loc]

    for epoch in range(args.epochs):
        # Training and validation using a full graph
        model.train()
        train_loss = []
        for step, batch in enumerate(train_loader):
            user, item, label = [_.to(device) for _ in batch]
            #logits = model.forward(g, user_key, item_key, user, item)
            logits = model.forward(homo_g, feats, user_target_idx[user], item_target_idx[item])

            # compute loss
            tr_loss = criterion(logits, label)
            train_loss.append(tr_loss)

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            #print("In epoch {}, step {}, Train Loss: {:.4f}".format(epoch, step, tr_loss))
        
        train_loss = np.sum(train_loss)

        model.eval()
        with torch.no_grad():
            validate_loss = []
            validate_acc = []
            for step, batch in enumerate(eval_loader):
                user, item, label = [_.to(device) for _ in batch]
                #logits = model.forward(g, user_key, item_key, user, item)
                logits = model.forward(homo_g, feats, user_target_idx[user], item_target_idx[item])

                # compute loss
                val_loss = criterion(logits, label)
                val_acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
                validate_loss.append(val_loss)
                validate_acc.append(val_acc)
        
            validate_loss = np.sum(validate_loss)
            validate_acc = np.mean(validate_acc)
        
            #validate
            if validate_acc > best_acc:
                best_acc = validate_acc
                best_epoch = epoch
                torch.save(model.state_dict(), 'TAHIN'+'_2_'+args.dataset)
                kill_cnt = 0
                print("saving model...")
            else:
                kill_cnt += 1
                if kill_cnt > args.early_stop:
                    print('early stop.')
                    break

            print("In epoch {}, Train Loss: {:.4f}, Valid Loss: {:.5}\n, Valid ACC: {:.5}".format(epoch, train_loss, validate_loss, validate_acc))
    
    #test use the best model
    model.eval()
    with torch.no_grad():
        #print('Load from epoch {}'.format(best_epoch))
        model.load_state_dict(torch.load('TAHIN'+'_2_'+args.dataset))
        test_loss = []
        test_acc = []
        test_auc = []
        test_f1 = []
        test_logloss = []
        for step, batch in enumerate(test_loader):
            user, item, label = [_.to(device) for _ in batch]
            #logits = model.forward(g, user_key, item_key, user, item)
            logits = model.forward(homo_g, feats, user_target_idx[user], item_target_idx[item])

            # compute loss
            loss = criterion(logits, label)
            acc = evaluate_acc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
            auc = evaluate_auc(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
            f1 = evaluate_f1_score(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
            log_loss = evaluate_logloss(logits.detach().cpu().numpy(), label.detach().cpu().numpy())
            
            test_loss.append(loss)
            test_acc.append(acc)
            test_auc.append(auc)
            test_f1.append(f1)
            test_logloss.append(log_loss)
        
        test_loss = np.sum(test_loss)
        test_acc = np.mean(test_acc)
        test_auc = np.mean(test_auc)
        test_f1 = np.mean(test_f1)
        test_logloss = np.mean(test_logloss)
        print("Test Loss: {:.5}\n, Test ACC: {:.5}\n, AUC: {:.5}\n, F1: {:.5}\n, Logloss: {:.5}\n".format(test_loss, test_acc, test_auc, test_f1, test_logloss))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='amazon', help='Dataset to use, default: amazon')
    parser.add_argument('--model', default='TAHIN', help='Model Name')

    parser.add_argument('--batch', default=128, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epochs', type=int, default=500, help='Maximum number of epochs')
    parser.add_argument('--wd', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of processes to construct batches')
    parser.add_argument('--seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('--early_stop', default=25, type=int, help='Patience for early stop.')
    
    parser.add_argument('--in_size', default=128, type=int, help='Initial dimension size for entities.')
    parser.add_argument('--hidden_size', default=128, type=int, help='Hidden dimension size for entities.')
    parser.add_argument('--out_size', default=128, type=int, help='Output dimension size for entities.')
    
    parser.add_argument('--layer_num_heads', default='[1]', nargs='?', help='Number of attention heads')
    parser.add_argument('--layer_dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout to use in the linear layer.')
    
    args = parser.parse_args()
    '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    '''
    print(args)

    args.layer_num_heads = eval(args.layer_num_heads)

    main(args)
