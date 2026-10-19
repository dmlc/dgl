"""
Configuration module for LightGCN.
Handles command-line argument parsing and configuration.
"""
import argparse


parser = argparse.ArgumentParser(description="Go lightGCN")
parser.add_argument('--batch', type=int,default=2048,
                    help="the batch size for bpr loss training procedure")
parser.add_argument('--recdim', type=int,default=16,
                    help="the embedding size of lightGCN")
parser.add_argument('--layer', type=int,default=1,
                    help="the layer num of lightGCN")
parser.add_argument('--lr', type=float,default=0.001,
                    help="the learning rate")
parser.add_argument('--decay', type=float,default=1e-4,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--testbatch', type=int,default=100,
                    help="the batch size of users for testing")
parser.add_argument('--dataset', type=str,default='gowalla',
                    help="available datasets: [gowalla, yelp2018, amazon-book]")
parser.add_argument('--path', type=str,default="./checkpoints",
                    help="path to save weights")
parser.add_argument('--topks', nargs='?',default=[20],
                    help="@k test list")
parser.add_argument('--load', type=int,default=0)
parser.add_argument('--epochs', type=int,default=100)
parser.add_argument('--device', type=str,default='cpu')
parser.add_argument('--seed', type=int, default=2020, help='random seed')

args = parser.parse_args()

# Print all configuration arguments
print("=" * 50)
print("Configuration Arguments:")
print("=" * 50)
for key, value in vars(args).items():
    print(f"  {key}: {value}")
print("=" * 50)