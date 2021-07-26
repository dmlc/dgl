# Implement EvolveGCN with DGL
paper link: [EvolveGCN](https://arxiv.org/abs/1902.10191)  
official code: [IBM/EvolveGCN](https://github.com/IBM/EvolveGCN)  
another implement: [pyG_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcno.py)  

## Dependency:
* dgl
* pandas
* numpy

## Run
* donwload Elliptic dataset from [kaggle](https://kaggle.com/ellipticco/elliptic-data-set)
* unzip the dataset into a raw directory, such as /home/Elliptic/elliptic_bitcoin_dataset/
* make a new dir to save processed data, such as /home/Elliptic/processed/  
* run train.py by:
```bash
python train.py --raw-dir /home/Elliptic/elliptic_bitcoin_dataset/ --processed-dir /home/Elliptic/processed/
```

## Result
Using EvolveGCN-O can match the results of Fig.3 and Fig.4 in the paper.
(May need to run several times to get the average)


## Attention:  
* Currently only the Elliptic dataset is used.
* EvolveGCN-H is not solid in Elliptic dataset, the official code is the same.   

Official code result when use EvolveGCN-H:  
1. set seed to 1234, finally result is :
> TEST epoch 189: TEST measures for class 1 - precision 0.3875 - recall 0.5714 - f1 0.4618  
2. not set seed manually, run the same code three times:
> TEST epoch 168: TEST measures for class 1 - precision 0.3189 - recall 0.0680 - f1 0.1121  
> TEST epoch 270: TEST measures for class 1 - precision 0.3517 - recall 0.3018 - f1 0.3249  
> TEST epoch 455: TEST measures for class 1 - precision 0.2271 - recall 0.2995 - f1 0.2583  
