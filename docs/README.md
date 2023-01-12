DGL document and tutorial folder
================================

Requirements
------------
You need to build DGL locally first (as described [here](https://docs.dgl.ai/install/index.html#install-from-source)), and ensure the following python packages are installed:
* sphinx==4.2.0
* sphinx-gallery
* sphinx_rtd_theme
* sphinx_copybutton
* torch
* mxnet
* pillow
* matplotlib


Build documents
---------------
First, clean up existing files:
```
./clean.sh
```

Then build:
```
make html
```

Note: due to the backend loading issue, it actually takes 2 rounds to build:
1. build tutorials that uses MXNet as backend
2. build tutorials that uses PyTorch as backend

Render locally
--------------
```
cd build/html
python3 -m http.server 8000
```

Add new folders
---------------
Add the path of the new folder in the two lists `examples_dirs` and `gallery_dirs` in docs/source/conf.py.
