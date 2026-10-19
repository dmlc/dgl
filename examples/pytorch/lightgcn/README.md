LightGCN使用DGL实现
============
参考论文：https://arxiv.org/pdf/2002.02126
论文提供的pytorch实现：https://github.com/gusye1234/LightGCN-PyTorch
-------

运行load_gowalla.sh以及load_amazon-book.sh以下载这两个数据集。yelp2018数据集已下载

示例运行：
python main.py --dataset gowalla --batch 2048 --recdim 64

python main.py --dataset amazon-book --batch 2048 --recdim 64

python main.py --dataset yelp2018 --batch 2048 --recdim 64

运行时间较长，需耐心等待若干分钟

优化代码，减少运行时长，可视化运行时长


