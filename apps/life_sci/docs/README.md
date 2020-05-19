Documentation and Tutorials
===

Requirements
------------
* sphinx
* sphinx-gallery
* sphinx_rtd_theme

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

Render locally
--------------
```
cd build/html
python3 -m http.server 8000
```
