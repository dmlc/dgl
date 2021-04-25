# **DGL Implementation of RECT (TKDE20)**

This DGL example implements the GNN model **RECT** (or more specifically its supervised part **RECT-L**) proposed in the paper [Network Embedding with Completely-imbalanced Labels](https://ieeexplore.ieee.org/document/8979355). The authors' original implementation can be found [here](https://github.com/zhengwang100/RECT).



## Example Implementor

This example was implemented by [Tingzhang Zhao](https://github.com/Fizyhsp) when he was an undergraduate at USTB.



## **Dataset and experimental setting**

Two DGL's build-in datasets (Cora and Citeseer) with their default train/val/test settings are used in this example. In addition, as this paper considers the zero-shot (i.e., completely-imbalanced) label setting, those "unseen" classes should be removed from the training set, as suggested in the paper. In this example, in each dataset, we simply remove the 2-3 classes (i.e., these 2-3 classes are unseen classes) from the labeled training set. Then, we obtain graph embedding results by different models. Finally, with the obtained embedding results and the original balanced labels, we train a logistic regression classifier to evaluate the model performance.



## **Usage** 

`python main.py --dataset cora --gpu 0 --model-opt RECT-L --removed-class 0 1 2` #reproducing the RECT-L on "cora" datasets in the zero-shot label setting using GPU

`python main.py --dataset cora --gpu 0 --model-opt GCN --removed-class 0 1 2` #reproducing the GCN on "cora" datasets in the zero-shot label setting using GPU

`python main.py --dataset cora --gpu 0 --model-opt NodeFeats --removed-class 0 1 2` # evaluating the original node features using GPU



## **Performance**

The performance results are are as follows:

| **Datasets/Models** | **NodeFeats** | **GCN** | **RECT-L** |
| :-----------------: | :-----------: | :-----: | :--------: |
|      **Cora**       |     47.56     |  51.26  | **68.60**  |
|    **Citeseer**     |     42.04     |  37.55  | **56.32**  |

<center>Table 1：node classification results with the first three classes as "unseen"</center>
<br/><br/>


| **Datasets/Models** | **NodeFeats** | **GCN** | **RECT-L** |
| :-----------------: | :-----------: | :-----: | :--------: |
|      **Cora**       |     47.56     |  56.91  | **69.30**  |
|    **Citeseer**     |     42.04     |  45.69  | **61.85**  |

<center>Table 2：node classification results with the last two classes as "unseen"</center>
<br/>
