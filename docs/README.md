DGL document and tutorial folder
================================

Requirements
------------
* sphinx
* sphinx-gallery
* sphinx_rtd_theme
* Both pytorch and mxnet installed.

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
