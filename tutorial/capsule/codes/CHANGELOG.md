# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2018-01-30
### Added
- Supports and works with CIFAR10 dataset.

### Changed
- Upgrade to PyTorch 0.3.0.
- Supports CUDA 9.
- Drop our custom softmax function and switch to PyTorch softmax function.
- Modify the save_image utils function to handle 3-channel (RGB) image.

### Fixed
- Compatibilities with PyTorch 0.3.0.

## [0.3.0] - 2017-11-27
### Added
- Decoder network PyTorch module.
- Reconstruct image with Decoder network during testing.
- Save the original and recontructed images into file system.
- Log the original and reconstructed images using TensorBoard.

### Changed
- Refactor reconstruction loss function and decoder network.
- Remove image reconstruction from training.

## [0.2.0] - 2017-11-26
### Added
- New dependencies for TensorBoard and tqdm.
- Logging losses and accuracies with TensorBoard.
- New utils functions for:
    - computing accuracy
    - convert values of the model parameters to numpy.array.
    - parsing boolean values with argparse
- Softmax function that takes a dimension.
- More detailed code comments.
- Show margin loss and reconstruction loss in logs.
- Show accuracy in train logs.

### Changed
- Refactor loss functions.
- Clean codes.

### Fixed
- Runtime error during pip install requirements.txt
- Bug in routing algorithm.

## [0.1.0] - 2017-11-12
### Added
- Implemented reconstruction loss.
- Saving reconstructed image as file.
- Improve training speed by using PyTorch DataParallel to wrap our model.
    - PyTorch will parallelized the model and data over multiple GPUs.
- Supports training:
    - on CPU (tested with macOS Sierra)
    - on one GPU (tested with 1 Tesla K80 GPU)
    - on multiple GPU (tested with 8 GPUs)
    - with or without CUDA (tested with CUDA version 8.0.61)
    - cuDNN 5 (tested with cuDNN 5.1.3)

### Changed
- More intuitive variable naming.

### Fixed
- Resolve Pylint warnings and reformat code.
- Missing square in equation 4 for margin (class) loss.

## 0.0.1 - 2017-11-04
### Added
- Initial release. The first beta version. API is stable. The code runs. So, I think it's safe to use for development but not ready for general production usage.

[Unreleased]: https://github.com/cedrickchee/capsule-net-pytorch/compare/v1.0.0...HEAD
[0.1.0]: https://github.com/cedrickchee/capsule-net-pytorch/compare/v0.0.1...v0.1.0
[0.2.0]: https://github.com/cedrickchee/capsule-net-pytorch/compare/v0.1.0...v0.2.0
[0.3.0]: https://github.com/cedrickchee/capsule-net-pytorch/compare/v0.2.0...v0.3.0
[0.4.0]: https://github.com/cedrickchee/capsule-net-pytorch/compare/v0.3.0...v0.4.0
