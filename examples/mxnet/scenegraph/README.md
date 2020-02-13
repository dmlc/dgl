# Scene Graph Extraction

Scene graph extraction aims at not only detect objects in the given image, but also classify the relationships between pairs of them.

This example reproduces [Graphical Contrastive Losses for Scene Graph Parsing](https://arxiv.org/abs/1903.02728), author's code can be found [here](https://github.com/NVIDIA/ContrastiveLosses4VRD).

## Results

**VisualGenome**

| Model     | Backbone  | mAP@50   | SGDET@20 | SGDET@50 | SGDET@100 | PHRCLS@20 | PHRCLS@50 |PHRCLS@100 | PREDCLS@20 | PREDCLS@50 | PREDCLS@100 |
| :---      | :---      | :---     | :---     | :---     | :---      | :---      | :---      | :---      | :---       | :---       | :---        |
| RelDN, L0 | ResNet101 | 29.5     | 22.65    | 30.02    | 35.04     | 32.84     | 35.60     | 36.26     | 60.58      | 65.53      | 66.51       |

## Dependencies

This implementation is based on GluonCV. Install GluonCV with 

```
pip install gluoncv --upgrade
```

## Data preparation

We provide scripts to download and prepare the VisualGenome dataset. One can run with

```
python data/prepare_visualgenome.py
```

## Object Detector

First one need to train the object detection model on VisualGenome.

```
bash train_faster_rcnn.sh
```

## Training RelDN

With a trained Faster R-CNN model, one can start the training of RelDN model by

```
bash train_reldn.sh
```

## Validate RelDN

After the training, one can evaluate the results with multiple commonly-used metrics:

```
bash validate_reldn.sh
```
