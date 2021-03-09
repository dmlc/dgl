from dataloader import GraphCollator
from models import OfflinePrepareLayer,InteractionGNN
from dataloader import TaichiTrainDataset,TaichiValidDataset,TaichiTestDataset
import torch
from torch.utils.data import DataLoader
from pyinstrument import Profiler
import argparse
import os

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

args = argparser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
num_workers = args.num_workers
PROFILE = args.profile
device = torch.device('cpu') if args.gpu==-1 else torch.device('cuda:{}'.format(args.gpu))


prep = OfflinePrepareLayer(train_dataset.n_particles,
                   batch_size,
                   train_dataset.dim,
                   torch.from_numpy(train_dataset.boundary).to(device),
                   {'vel_mean':0,'vel_std':1},
                   0.04).to(device)

ignn = InteractionGNN(10,
                      14,
                      3,
                      20,
                      10,
                      2,
                      2).to(device)

collator = GraphCollator(radius = 0.03)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,collate_fn=collator)
valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=collator)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=collator)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ignn.parameters(),lr=0.01)


# Batch might be problematic for model especially for input
def train():
    torch.multiprocessing.freeze_support()
    if PROFILE:
        profiler = Profiler()
        profiler.start()
    total_loss = 0
    for i,(src_batch_g,src_coord,src_vels,dst_coord) in enumerate(train_dataloader):
        optimizer.zero_grad()
        src_batch_g = src_batch_g.to(device)
        src_coord = src_coord.to(device)
        src_vels  = src_vels.to(device)
        dst_coord = dst_coord.to(device)
        node_feature,edge_feature,current_v = prep(src_batch_g,src_coord,src_vels)
        # Forward interaction network for prediction
        # Manually add acceleration
        pred_acc = ignn(src_batch_g,node_feature,edge_feature) - 9.8
        # Done by semi implicite Euler
        pred_v   = current_v + pred_acc*substeps*dt
        pred_x   = src_coord + (pred_v+current_v)/2*substeps*dt
        loss = loss_fn(pred_x,dst_coord)
        total_loss = total_loss*i/(i+1) +1/(i+1)*float(loss)
        loss.backward()
        optimizer.step()
        print("Batch {}".format(i))
        if PROFILE:
            profiler.stop()
            profiler.start()
            print(profiler.output_text(unicode=True,color=True))
    return total_loss

def evaluate(loader):
    total_loss = 0
    for i,(src_batch_g,src_coord,src_vels,dst_coord) in enumerate(loader):
        src_batch_g = src_batch_g.to(device)
        src_coord = src_coord.to(device)
        src_vels  = src_vels.to(device)
        dst_coord = dst_coord.to(device)
        node_feature,edge_feature,current_v = prep(src_batch_g,src_coord,src_vels)
        # Forward interaction network for prediction
        # Manually add acceleration
        pred_acc = ignn(src_batch_g,node_feature,edge_feature) - 9.8
        # Done by semi implicite Euler
        pred_v   = current_v + pred_acc*substeps*dt
        pred_x   = src_coord + (pred_v+current_v)/2*substeps*dt
        loss = loss_fn(pred_x,dst_coord)
        total_loss = total_loss*i/(i+1) +1/(i+1)*float(loss)
        print("Batch {}".format(i))
    return total_loss

if __name__ == '__main__':
    epochs = 20
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')
    for i in range(epochs):
        train_loss = train()
        valid_loss = evaluate(valid_dataloader)
        test_loss  = evaluate(test_dataloader)
        print("Epoch {} Valid Loss: {} Test Loss: {}".format(epochs,valid_loss,test_loss))
        torch.save(ignn.state_dict(),'saved_models/model{}.pth'.format(i))

    



        
        
