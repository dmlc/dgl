# DGL Implementation of EEG-GCNN Paper
This example is a simplified version that presents how to utilize the original EEG-GCNN model proposed in the paper [EEG-GCNN](http://proceedings.mlr.press/v136/wagh20a.html), implemented with DGL library. The example removes cross validation and optimal decision boundary that are used in the original code. The performance stats are slightly different from what is present in the paper. The original code is [here](https://github.com/neerajwagh/eeg-gcnn).

## All References
- [ML4H Poster](https://drive.google.com/file/d/14nuAQKiIud3p6-c8r9WLV2tAvCyRwRev/view?usp=sharing) can be helpful for understanding data preprocessing, model, and performance of the project. 
- The recording of presentation by the author Neeraj Wagh can be found on [slideslive](https://slideslive.com/38941020/eeggcnn-augmenting-electroencephalogrambased-neurological-disease-diagnosis-using-a-domainguided-graph-convolutional-neural-network?ref=account-folder-62123-folders).
- The slides used during the presentation can be found [here](https://drive.google.com/file/d/1dXT4QAUXKauf7CAkhrVyhR2PFUsNh4b8/view?usp=sharing).
- Raw Data can be found with these two links: [MPI LEMON](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) (no registration needed), [TUH EEG Abnormal Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/) ([needs registration](https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php))

## Dependencies

- Python 3.8.1
- PyTorch 1.7.0
- DGL 0.6.1
- numpy 1.20.2
- Sklearn 0.24.2
- pandas 1.2.4
## Dataset
- Final Models, Pre-computed Features, Training Metadata can be downloaded through [FigShare](https://figshare.com/articles/software/EEG-GCNN_Supporting_Resources_for_Reproducibility/13251452).
- In ```EEGGraphDataset.py```, we specify the channels and electrodes and use precomputed spectral coherence values to compute the edge weights. To use this example in your own advantage, please specify your channels and electrodes in ```__init__``` function of ```EEGGraphDataset.py```.
- To generate spectral coherence values, please refer to [spectral_connectivity](https://mne.tools/stable/generated/mne.connectivity.spectral_connectivity.html) function in mne library. An example usage may take the following form:
```python
     # ....loop over all windows in dataset....

        # window data is 10-second preprocessed multi-channel timeseries (shape: n_channels x n_timepoints) containing all channels in ch_names
        window_data = np.expand_dims(window_data, axis=0)

        # ch_names are listed in EEGGraphDataset.py
        for ch_idx, ch in enumerate(ch_names):
            # number of channels is is len(ch_names), which is 8 in our case.
            spec_coh_values, _, _, _, _ = mne.connectivity.spectral_connectivity(data=window_data, method='coh', indices=([ch_idx]*8, range(8)), sfreq=SAMPLING_FREQ,
                                              fmin=1.0, fmax=40.0, faverage=True, verbose=False)
```
## How to Run
- First, download ```figshare_upload/master_metadata_index.csv```, ```figshare_upload/psd_features_data_X```, ```figshare_upload/labels_y```, ```figshare_upload/psd_shallow_eeg-gcnn/spec_coh_values.npy```, and ```figshare_upload/psd_shallow_eeg-gcnn/standard_1010.tsv.txt```. Put them in the repo. <br>
- You may download these files by running:
```python
wget https://ndownloader.figshare.com/files/25518170
```
- You will need to unzip the downloaded file.
- Then run: 
```python
python main.py
```
- The default model used is ```shallow_EEGGraphConvNet.py```. To use ```deep_EEGGraphConvNet.py```, run:
```python
python main.py --model deep
```
- After the code executes, you will be able to see similar stats in performance section printed. The code will save the trained model from every epoch.
## Performance

|      DGL          | AUC         | Bal. Accuracy |
|-------------------|-------------|---------------|
| Shallow EEG-GCNN  | 0.832       | 0.750         |
| Deep EEG-GCNN     | 0.830       | 0.736         |

Shallow_EEGGraphConvNet    |              AUC          |     Bal.Accuracy      |
:-------------------------:|:-------------------------:|:---------------------:|
![shallow_loss](https://user-images.githubusercontent.com/53772888/128595442-d185bd74-5c5d-4118-a6b7-b89dd307d3aa.png)  |![shallow_auc](https://user-images.githubusercontent.com/53772888/128595453-2f3b181a-bcb7-4da4-becd-7a7aa62083bc.png)|![shallow_bacc](https://user-images.githubusercontent.com/53772888/128595456-b293c888-bf8c-4f37-bd58-d01885da3832.png)

Deep_EEGGraphConvNet            |  AUC | Bal.Accuracy |
:-------------------------:|:-------------------------:|:---------------:|
![deep_loss](https://user-images.githubusercontent.com/53772888/128595458-e4a76591-11cf-405f-9c20-2d161e49c358.png)|![deep_auc](https://user-images.githubusercontent.com/53772888/128595462-7a7bfb67-4601-4e83-8764-d7c44bf979b5.png)|![deep_bacc](https://user-images.githubusercontent.com/53772888/128595467-1a0cd37d-0152-431b-a29b-a40bafb71be5.png)

### Contact

- Email to John(_wei33@illinois.edu_)
- You may also contact the authors:
  - Neeraj: nwagh2@illinois.edu / [Website](http://neerajwagh.com/) / [Twitter](https://twitter.com/neeraj_wagh) / [Google Scholar](https://scholar.google.com/citations?hl=en&user=lCy5VsUAAAAJ)
  - Yoga: varatha2@illinois.edu / [Website](https://sites.google.com/view/yoga-personal/home) / [Google Scholar](https://scholar.google.com/citations?user=XwL4dBgAAAAJ&hl=en)

### Citation

Wagh, N. & Varatharajah, Y.. (2020). EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network. Proceedings of the Machine Learning for Health NeurIPS Workshop, in PMLR 136:367-378 Available from http://proceedings.mlr.press/v136/wagh20a.html.
