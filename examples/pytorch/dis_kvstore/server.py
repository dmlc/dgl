# This is a simple MXNet server demo shows how to use DGL distributed kvstore.
import dgl
import argparse
import torch as th

ndata_g2l = []
edata_g2l = []

ndata_g2l.append({'ndata':th.tensor([0,1,0,0,0,0,0,0])})
ndata_g2l.append({'ndata':th.tensor([0,0,0,1,0,0,0,0])})
ndata_g2l.append({'ndata':th.tensor([0,0,0,0,0,1,0,0])})
ndata_g2l.append({'ndata':th.tensor([0,0,0,0,0,0,0,1])})

edata_g2l.append({'edata':th.tensor([0,1,0,0,0,0,0,0])})
edata_g2l.append({'edata':th.tensor([0,0,0,1,0,0,0,0])})
edata_g2l.append({'edata':th.tensor([0,0,0,0,0,1,0,0])})
edata_g2l.append({'edata':th.tensor([0,0,0,0,0,0,0,1])})

def start_server(args):
    
    dgl.contrib.start_server(
        server_id=args.id,
        ip_config='ip_config.txt',
        num_client=4,
        ndata={'ndata':th.tensor([[0.,0.,0.],[0.,0.,0.]])},
        edata={'edata':th.tensor([[0.,0.,0.],[0.,0.,0.]])},
        ndata_g2l=ndata_g2l[args.id],
        edata_g2l=edata_g2l[args.id])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kvstore')
    parser.add_argument("--id", type=int, default=0, help="node ID")
    args = parser.parse_args()

    start_server(args)