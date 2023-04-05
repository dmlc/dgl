DGL document and tutorial folder
================================

Requirements
------------
You need to build DGL locally first (as described [here](https://docs.dgl.ai/install/index.html#install-from-source)), and ensure the following python packages are installed:

* sphinx==4.2.0
* sphinx-gallery
* sphinx_rtd_theme
* sphinx_copybutton
* nbsphinx>=0.8.11
* nbsphinx-link>=1.3.0
* pillow
* matplotlib
* nltk
* seaborn
* ogb
* rdflib


Build documents
---------------
First, clean up existing files:
```
./clean.sh
```

To build for PyTorch backend only,
```
make pytorch
```

To build for MXNet backend only,
```
make mxnet
```

To build for both backends,
```
make html
```

Render locally
--------------
```
cd build/html
python3 -m http.server 8000
```

Add new folders
---------------
Add the path of the new folder in the two lists `examples_dirs` and `gallery_dirs` in docs/source/conf.py.
