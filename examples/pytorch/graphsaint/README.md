# GraphSaint
The dgl implementation of GraphSaint
# comparison
## F1-micro
| Method | PPI | Flickr | Reddit |
| --- | --- | --- | --- |
| Node(paper) | 0.960±0.001 | 0.507±0.001 | 0.962±0.001 |
| Edge(paper) | 0.981±0.007 | 0.510±0.002 | 0.966±0.001 |
| RW(paper) | 0.981±0.004 | 0.511±0.001 | 0.966±0.001 |
| Node(run) | 0.9628 | 0.5077 | 0.9622 |
| Edge(run) | 0.9810 | 0.5066 | 0.9656 |
| RW(run) | 0.9812 | 0.5104 | 0.9648 |
| Node(dgl) | 0.5462 | 0.4981 | 0.9373 |
| Edge(dgl) | 0.6065 | 0.5067 | 0.9137 |
| RW(dgl) | 0.4116 | nan | nan |

## Sample time
| Method | PPI | Flickr | Reddit |
| --- | --- | --- | --- |
| Node(run) | 1.0139 | 0.9574 | 9.0769 |
| Edge(run) | 0.8712 | 0.8764 | 4.7546 |
| RW(run) | 1.0880 | 1.7588 | 7.2055 |
| Node(dgl) | 0.8813 | 0.9633 | 70.0391 |
| Edge(dgl) | 0.8635 | 1.0568 | 87.0523 |
| RW(dgl) | 0.4818 | 0.7259 | 53.7538 |


## Problems
* GCN的结构在有些数据集上影响特别大，例如在Reddit上，hidden_dim分别设置为128和256，最终的结果会相差30个点左右。
