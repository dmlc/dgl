# Scene Graph Extraction

Scene graph extraction aims at not only detect objects in the given image, but also classify the relationships between pairs of them.

This example reproduces [Graphical Contrastive Losses for Scene Graph Parsing](https://arxiv.org/abs/1903.02728), author's code can be found [here](https://github.com/NVIDIA/ContrastiveLosses4VRD).

![DEMO](https://raw.githubusercontent.com/dmlc/web-data/master/dgl/examples/mxnet/scenegraph/old-couple-pred.png)

## Results

**VisualGenome**

| Model     | Backbone  | mAP@50   | SGDET@20 | SGDET@50 | SGDET@100 | PHRCLS@20 | PHRCLS@50 |PHRCLS@100 | PREDCLS@20 | PREDCLS@50 | PREDCLS@100 |
| :---      | :---      | :---     | :---     | :---     | :---      | :---      | :---      | :---      | :---       | :---       | :---        |
| RelDN, L0 | ResNet101 | 29.5     | 22.65    | 30.02    | 35.04     | 32.84     | 35.60     | 36.26     | 60.58      | 65.53      | 66.51       |

## Preparation

This implementation is based on GluonCV. Install GluonCV with 

```
pip install gluoncv --upgrade
```

The implementation contains the following files:

```
.
|-- data
|   |-- dataloader.py
|   |-- __init__.py
|   |-- object.py
|   |-- prepare_visualgenome.py
|   `-- relation.py
|-- demo_reldn.py
|-- model
|   |-- faster_rcnn.py
|   |-- __init__.py
|   `-- reldn.py
|-- README.md
|-- train_faster_rcnn.py
|-- train_faster_rcnn.sh
|-- train_freq_prior.py
|-- train_reldn.py
|-- train_reldn.sh
|-- utils
|   |-- build_graph.py
|   |-- __init__.py
|   |-- metric.py
|   |-- sampling.py
|   `-- viz.py
|-- validate_reldn.py
`-- validate_reldn.sh
```

- The folder `data` contains the data preparation script, and definition of datasets for object detection and scene graph extraction.
- The folder `model` contains model definition.
- The folder `utils` contains helper functions for training, validation, and visualization.
- The script `train_faster_rcnn.py` trains a Faster R-CNN model on VisualGenome dataset, and `train_faster_rcnn.sh` includes preset parameters.
- The script `train_freq_prior.py` trains the frequency counts for RelDN model training.
- The script `train_reldn.py` trains a RelDN model, and `train_reldn.sh` includes preset parameters.
- The script `validate_reldn.py` validate the trained Faster R-CNN and RelDN models, and `validate_reldn.sh` includes preset parameters.
- The script `demo_reldh.py` makes use of trained parameters and extract an scene graph from an arbitrary input image.

Below are further steps on training your own models. Besides, we also provide pretrained model files for validation and demo:

1. [Faster R-CNN Model for Object Detection](http://dgl-data/models/SceneGraph/faster_rcnn_resnet101_v1d_visualgenome.params)
2. [RelDN Model](http://dgl-data/models/SceneGraph/reldn.params)
3. [Faster R-CNN Model for Edge Feature](http://dgl-data/models/SceneGraph/detector_feature.params)

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

It runs for about 20 hours on a machine with 64 CPU cores and 8 V100 GPUs.

## Training RelDN

With a trained Faster R-CNN model, one can start the training of RelDN model by

```
bash train_reldn.sh
```

It runs for about 2 days with one single GPU and 8 CPU cores.

## Validate RelDN

After the training, one can evaluate the results with multiple commonly-used metrics:

```
bash validate_reldn.sh
```

## Demo

We provide a demo script of running the model with real-world pictures. Be aware that you need trained model to generate meaningful results from the demo, otherwise the script will download the pre-trained model automatically.
