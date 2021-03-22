from dataloader import GraphCollator
from models import OfflinePrepareLayer,InteractionGNN
from dataloader import TaichiTrainDataset,TaichiValidDataset,TaichiTestDataset
import torch
from torch.utils.data import DataLoader
from pyinstrument import Profiler
import argparse
import os
from utils import normalize_acc
import time

train_dataset = TaichiTrainDataset()
valid_dataset = TaichiValidDataset()
test_dataset  = TaichiTestDataset()

dt = train_dataset.dt
substeps = train_dataset.substeps

argparser = argparse.ArgumentParser()
argparser.add_argument("--batch_size",type=int,default=16)
argparser.add_argument("--epochs",type=int,default=20)
argparser.add_argument("--num_workers",type=int,default=0)
argparser.add_argument("--gpu",type=int,default=-1)
argparser.add_argument("--profile",action='store_true',default=False)
argparser.add_argument("--radius",type=float,default=0.015)
argparser.add_argument("--lr",type=float,default=1e-4)
argparser.add_argument("--load_model",type=str,default='None')

args = argparser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
num_workers = args.num_workers
PROFILE = args.profile
device = torch.device('cpu') if args.gpu==-1 else torch.device('cuda:{}'.format(args.gpu))
radius = args.radius
noise_scale = 0.0003

prep = OfflinePrepareLayer(train_dataset.n_particles,
                   batch_size,
                   train_dataset.dim,
                   torch.from_numpy(train_dataset.boundary).to(device),
                   {'vel_mean':train_dataset.vel_mean.to(device),
                    'vel_std':train_dataset.vel_std.to(device)},
                   radius,
                   noise_scale).to(device)

ignn = InteractionGNN(10,
                      14,
                      3,
                      20,
                      10,
                      2,
                      2).to(device)
if args.load_model != 'None':
    ignn.load_state_dict(torch.load('saved_models/{}'.format(args.load_model)))

collator = GraphCollator(radius = radius)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,collate_fn=collator)
valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=collator)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=collator)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ignn.parameters(),lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.99)


# Batch might be problematic for model especially for input
def train():
    torch.multiprocessing.freeze_support()
    if PROFILE:
        profiler = Profiler()
        profiler.start()
    total_loss = 0
    total_evalloss = 0
    for i,(src_batch_g,src_coord,src_vels,dst_coord,dst_acc) in enumerate(train_dataloader):
        optimizer.zero_grad()
        src_batch_g = src_batch_g.to(device)
        src_coord = src_coord.to(device)
        src_vels  = src_vels.to(device)
        dst_coord = dst_coord.to(device)
        dst_acc = dst_acc.float()
        dst_acc = dst_acc.to(device)
        node_feature,edge_feature,current_v = prep(src_batch_g,src_coord,src_vels)
        # Forward interaction network for prediction
        # Manually add acceleration
        pred_acc = ignn(src_batch_g,node_feature,edge_feature)
        #pred_acc[:,1] = pred_acc[:,1]-9.8
        # Done by semi implicite Euler
        #pred_v   = current_v + pred_acc*substeps*dt
        pred_v = current_v + pred_acc
        #pred_x   = src_coord + (pred_v+current_v)/2*substeps*dt
        pred_x = src_coord + pred_v
        pred_acc = normalize_acc(pred_acc,train_dataset.acc_mean,train_dataset.acc_std)
        dst_acc  = normalize_acc(pred_acc,train_dataset.acc_mean,train_dataset.acc_std)
        loss = loss_fn(pred_acc,dst_acc)
        loss_eval = loss_fn(pred_x,dst_coord)
        total_loss = total_loss*i/(i+1) +1/(i+1)*float(loss)
        total_evalloss = total_evalloss*i/(i+1)+1/(i+1)*float(loss_eval)
        loss = loss + total_evalloss
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            #print("Batch {}".format(i))
            scheduler.step()
        if PROFILE:
            profiler.stop()
            profiler.start()
            print(profiler.output_text(unicode=True,color=True))
    return total_loss,total_evalloss

def evaluate(loader):
    total_loss = 0
    for i,(src_batch_g,src_coord,src_vels,dst_coord,_) in enumerate(loader):
        src_batch_g = src_batch_g.to(device)
        src_coord = src_coord.to(device)
        src_vels  = src_vels.to(device)
        dst_coord = dst_coord.to(device)
        node_feature,edge_feature,current_v = prep(src_batch_g,src_coord,src_vels)
        # Forward interaction network for prediction
        # Manually add acceleration
        #pred_acc = ignn(src_batch_g,node_feature,edge_feature) - 9.8
        pred_acc = ignn(src_batch_g,node_feature,edge_feature)
        # Done by semi implicite Euler
        #pred_v   = current_v + pred_acc*substeps*dt
        pred_v = current_v + pred_acc
        #pred_x   = src_coord + (pred_v+current_v)/2*substeps*dt
        pred_x = src_coord + pred_v
        loss = loss_fn(pred_x,dst_coord)
        total_loss = total_loss*i/(i+1) +1/(i+1)*float(loss)
        '''
        if i % 100 == 0:
            print("Batch {}".format(i))
        '''
        
    return total_loss

if __name__ == '__main__':
    epochs = args.epochs
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')
    with open("logging.txt",'w') as logfile:
        for i in range(epochs):
            train_loss,train_evalloss = train()
            valid_loss = evaluate(valid_dataloader)
            test_loss  = evaluate(test_dataloader)
            log_text = "Epoch {} Valid Loss: {} Test Loss: {} {}".format(i,valid_loss,test_loss,time.ctime())
            print(log_text)
            logfile.writelines(log_text+'\n')
            torch.save(ignn.state_dict(),'saved_models/model{}.pth'.format(i))

    



        
        
