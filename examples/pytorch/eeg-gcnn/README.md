# DGL Implementation of EEG-GCNN Paper
This example is a simplified version that presents how to utilize the original EEG-GCNN model proposed in the paper [EEG-GCNN](https://arxiv.org/abs/2011.12107), implemented with DGL library. The original code is [here](https://github.com/neerajwagh/eeg-gcnn).

## All References
- Paper can also be found on [PMLR](http://proceedings.mlr.press/v136/wagh20a.html).
- [ML4H Poster](https://drive.google.com/file/d/14nuAQKiIud3p6-c8r9WLV2tAvCyRwRev/view?usp=sharing) can be helpful for understanding data preprocessing, model, and performance of the project. 
- The recording of presentation by the author Neeraj Wagh can be found on [slideslive](https://slideslive.com/38941020/eeggcnn-augmenting-electroencephalogrambased-neurological-disease-diagnosis-using-a-domainguided-graph-convolutional-neural-network?ref=account-folder-62123-folders).
- The slides used during the presentation can be found [here](https://drive.google.com/file/d/1dXT4QAUXKauf7CAkhrVyhR2PFUsNh4b8/view?usp=sharing).
- Raw Data can be found in these two links: [MPI LEMON](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) (no registration needed), [TUH EEG Abnormal Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/) ([needs registration](https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php))

## Dependencies

- Python 3.8.1
- PyTorch 1.7.0
- DGL 0.6.1
- numpy 1.20.2
- Sklearn 0.24.2

## Dataset
- Final Models, Pre-computed Features, Training Metadata can be downloaded through [FigShare](https://figshare.com/articles/software/EEG-GCNN_Supporting_Resources_for_Reproducibility/13251452).
- In EEGGraphDataset.py, we specify the channels and electrodes and use precomputed spectral coherence values to compute the edge weights. To use this example in your own advantage, please specify your channels, electrodes and generate your own spectral coherence values. All example datasets can be downloaded from FigShare.
## How to Run
First, download the precomputed data, labels, indices and put them in the repo. <br>
Then run 
```python
python main.py
```

## Performance
|Pytorch_geometric | AUC          | Precision     | Recall       | F-1          | Bal. Accuracy |
|------------------|--------------|---------------|--------------|--------------|---------------|
| Shallow EEG-GCNN | 0.867(0.005) | 0.985(0.004)  | 0.680(0.023) | 0.804(0.015) | 0.802(0.006)  |
| Deep EEG-GCNN    | 0.908(0.002) | 0.987(0.0001) | 0.753(0.007) | 0.855(0.005) | 0.842(0.004)  |

|      DGL          | AUC         | Precision     | Recall       | F-1          | Bal. Accuracy |
|-------------------|-------------|---------------|--------------|--------------|---------------|
| Shallow EEG-GCNN  | 0.875(0.036)| 0.980(0.013)  | 0.735(0.055) | 0.839(0.035) | 0.811(0.035)  |
| Deep EEG-GCNN     | 0.890(0.004)| 0.988(0.004)  | 0.723(0.035) | 0.834(0.022) | 0.829(0.005)  |

### Contact

- Email to John(_wei33@illinois.edu_)
- You may also contact the authors:
  - Neeraj: nwagh2@illinois.edu / [Website](http://neerajwagh.com/) / [Twitter](https://twitter.com/neeraj_wagh) / [Google Scholar](https://scholar.google.com/citations?hl=en&user=lCy5VsUAAAAJ)
  - Yoga: varatha2@illinois.edu / [Website](https://sites.google.com/view/yoga-personal/home) / [Google Scholar](https://scholar.google.com/citations?user=XwL4dBgAAAAJ&hl=en)

### Citation

Wagh, N. & Varatharajah, Y.. (2020). EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network. Proceedings of the Machine Learning for Health NeurIPS Workshop, in PMLR 136:367-378 Available from http://proceedings.mlr.press/v136/wagh20a.html.
