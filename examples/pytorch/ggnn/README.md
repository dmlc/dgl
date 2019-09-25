# Gated Graph Neural Network (GGNN)

- Paper link: https://arxiv.org/pdf/1511.05493.pdf

## Dependencies
- PyTorch 1.0+

- DGL 0.3.1+

## Solving bAbI tasks

- bAbI dataset is generated randomly by [script](https://github.com/facebook/bAbI-tasks). For convenience, we have generated the dataset we used in this example [here](https://s3.us-east-2.amazonaws.com/dgl.ai/models/ggnn_babi_data.zip). Unzip the file to current directory.

- For task 4, there are four question types: n, s, w, e, their corresponding question id are ranged from 0 to 3. For each question type, run the following commands.

```bash
python train_ns.py --task_id=4 --question_id=0 --train_num=50 --epochs=10
python train_ns.py --task_id=4 --question_id=1 --train_num=50 --epochs=10
python train_ns.py --task_id=4 --question_id=2 --train_num=50 --epochs=10
python train_ns.py --task_id=4 --question_id=3 --train_num=50 --epochs=10
```
The accuracies on 10 different test datasets are all averaged 100.0 with 0.0 std.

- For task 15, it has one quesiton type: has_fear, and it has question id 1, run the following 

```bash
python train_ns.py --task_id=15 --question_id=1 --train_num=50 --epochs=15 --lr=1e-2
```

We can get average accuray of 100.0 with std 0.0 on 10 different test datasets.

- For task 16, it has one quesiton type: has_color, and it has question id 1, run the following 

```bash
python train_ns.py --task_id=16 --question_id=1 --train_num=50 --epochs=20 --lr=1e-2
```

We can get average accuray of 100.0 with std 0.0 on 10 different test datasets.

- For task 18, there are two question types: >, <, which has question ids: 0 and 1. Run the following

```bash
python train_gc.py --task_id=18 --question_id=0 --train_num=50 --batch_size=10 --lr=1e-3 --epochs=20
python train_gc.py --task_id=18 --question_id=1 --train_num=50 --batch_size=10 --lr=1e-3 --epochs=20
```

We can get average accuracy of 100.0 with std 0.0 on 10 different test datasets for both question types.

- For task 19, the *Path Finding* task, there are no question types. Run the following command

```bash
python train_path_finding.py --train_num=250 --epochs=200
```

We can get average accuray of 97.8 with std 0.02 on 10 different test datasets.







