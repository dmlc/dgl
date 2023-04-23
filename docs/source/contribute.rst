Contribute to DGL
=================

Any contribution to DGL is welcome. This guide covers everything
about how to contribute to DGL.

General development process
---------------------------

A non-inclusive list of types of contribution is as follows:

* New features and enhancements (`example <https://github.com/dmlc/dgl/pull/331>`__).
* New NN Modules (`example <https://github.com/dmlc/dgl/pull/788>`__).
* Bugfix (`example <https://github.com/dmlc/dgl/pull/247>`__).
* Document improvement (`example <https://github.com/dmlc/dgl/pull/263>`__).
* New models and examples (`example <https://github.com/dmlc/dgl/pull/279>`__).

For features and bugfix, we recommend first raise an `issue <https://github.com/dmlc/dgl/issues>`__
using the corresponding issue template, so that the change could be fully discussed with
the community before implementation. For document improvement and new models, we suggest
post a thread in our `discussion forum <https://discuss.dgl.ai>`__.

Before development, please first read the following sections about coding styles and testing.
All the changes need to be reviewed in the form of `pull request <https://github.com/dmlc/dgl/pulls>`__.
Our `committors <https://github.com/orgs/dmlc/teams/dgl-team/members>`__
(who have write permission on the repository) will review the codes and suggest the necessary
changes. The PR could be merged once the reviewers approve the changes.

Git setup (for developers)
--------------------------

First, fork the DGL github repository. Suppose the forked repo is ``https://github.com/username/dgl``.

Clone your forked repository locally:

.. code-block:: bash

   git clone --recursive https://github.com/username/dgl.git


Setup the upstream to the DGL official repository:

.. code-block:: bash

   git remote add upstream https://github.com/dmlc/dgl.git

You could verify the remote setting by typing ``git remote -v``:

.. code-block:: bash

   origin  https://github.com/username/dgl.git (fetch)
   origin  https://github.com/username/dgl.git (push)
   upstream        https://github.com/dmlc/dgl.git (fetch)
   upstream        https://github.com/dmlc/dgl.git (push)

During developing, we suggest work on another branch than the master.

.. code-block:: bash

   git branch working-branch
   git checkout working-branch

Once the changes are done, `create a pull request <https://help.github.com/articles/creating-a-pull-request/>`__
so we could review your codes.

Once the pull request is merged, update your forked repository and delete your working branch:

.. code-block:: bash

   git checkout master
   git pull upstream master
   git push origin master  # update your forked repo
   git branch -D working-branch  # the local branch could be deleted

Coding styles
-------------

For python codes, we generally follow the `PEP8 style guide <https://www.python.org/dev/peps/pep-0008/>`__.
The python comments follow `NumPy style python docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__.

For C++ codes, we generally follow the `Google C++ style guide <https://google.github.io/styleguide/cppguide.html>`__.
The C++ comments should be `Doxygen compatible <http://www.doxygen.nl/manual/docblocks.html#cppblock>`__.

Coding styles check is mandatory for every pull requests. To ease the development, please check it
locally first (require cpplint and pylint to be installed first):

.. code-block:: bash

   bash tests/scripts/task_lint.sh

The python code style configure file is ``tests/lint/pylintrc``. We tweak it a little bit from
the standard. For example, following variable names are accepted:

* ``i,j,k``: for loop variables
* ``u,v``: for representing nodes
* ``e``: for representing edges
* ``g``: for representing graph
* ``fn``: for representing functions
* ``n,m``: for representing sizes
* ``w,x,y``: for representing weight, input, output tensors
* ``_``: for unused variables

Contributing New Models as Examples
-----------------------------------

To contribute a new model within a specific supported tensor framework (e.g. PyTorch, or MXNet), simply

1. Make a directory with the name of your model (say ``awesome-gnn``) within the directory
   ``examples/${DGLBACKEND}`` where ``${DGLBACKEND}`` refers to the framework name.
   
2. Populate it with your work, along with a README.  Make a pull request once you are done.  Your README should contain at least these:

   * Instructions for running your program.
   
   * The performance results, such as speed or accuracy or any metric, along with comparisons against some alternative implementations (if available).
   
     * Your performance metric does not have to beat others' implementation; they are just a signal of your code being *likely* correct.
     
     * Your speed also does not have to surpass others'.
     
     * However, better numbers are always welcomed.
   
3. The committers will review it, suggesting or making changes as necessary.

4. Resolve the suggestions and reviews, and go back to step 3 until approved.

5. Merge it and enjoy your day.

Data hosting
````````````

One often wishes to upload a dataset when contributing a new runnable model example, especially when covering
a new field not in our existing examples.

Uploading data file into the Git repository directly is a **bad idea** because we do not want the cloners to
always download the dataset no matter what.  Instead, we strongly suggest the data files be hosted on a
permanent cloud storage service (e.g. DropBox, Amazon S3, Baidu, Google Drive, etc.).

One can either

* Make your scripts automatically download your data if possible (e.g. when using Amazon S3), or
* Clearly state the instructions of downloading your dataset (e.g. when using Baidu, where auto-downloading
  is hard).
  
If you have trouble doing so (e.g. you cannot find a permanent cloud storage), feel free to post in our
`discussion forum <https://discuss.dgl.ai>`__.

Depending on the commonality of the contributed task, model, or dataset, we (the DGL team) would migrate
your dataset to the official DGL Dataset Repository on Amazon S3.  If you wish to host a particular dataset,
you can either

* DIY: make changes in the ``dgl.data`` module; see our :ref:`dataset APIs <apidata>` for more details, or,
* Post in our `discussion forum <https://discuss.dgl.ai>`__ (again).

Currently, all the datasets of DGL model examples are hosted on Amazon S3.

Contributing Core Features
--------------------------

We call a feature that goes into the Python ``dgl`` package a *core feature*.

Since DGL supports multiple tensor frameworks, contributing a core feature is no easy job.  However, we do
**NOT** require knowledge of all tensor frameworks.  Instead,

1. Before making a pull request, please make sure your code is covered with unit tests on **at least one**
   supported frameworks; see the `Building and Testing`_ section for details.
2. Once you have done that, make a pull request and summarize your changes, and wait for the CI to finish.
3. If the CI fails on a tensor platform that you are unfamiliar with (which is well often the case), please
   refer to `Supporting Multiple Platforms`_ section.
4. The committers will review it, suggesting or making changes as necessary.
5. Resolve the suggestions and reviews, and go back to step 3 until approved.
6. Merge it and enjoy your day.

Supporting Multiple Platforms
`````````````````````````````

This is the hard one, but you don't have to know PyTorch AND MXNet (maybe AND Tensorflow, AND Chainer, etc.,
in the future) to do so.  The rule of thumb in supporting Multiple Platforms is simple:

* In the ``dgl`` Python package, **always** avoid using framework-specific operators (*including array indexing!*)
  directly.  Use the wrappers in ``dgl.backend`` or ``numpy`` arrays instead.
* If you have trouble doing so (either because ``dgl.backend`` does not cover the necessary operator, or you don't
  have a GPU, or for whatever reason), please label your PR with the ``backend support`` tag, and one or more DGL
  team member who understand CPU AND GPU AND PyTorch AND MXNet (AND Tensorflow AND Chainer AND etc.) will
  look into it.

Building and Testing
````````````````````

To build DGL locally, follow the steps described in :ref:`Install from source <install-from-source>`.
However, to ease the development, we suggest NOT install DGL but directly working in the source tree.
To achieve this, export following environment variables:

.. code-block:: bash

   export DGL_HOME=/path/to/your/dgl/clone
   export DGL_LIBRARY_PATH=$DGL_HOME/build
   export PYTHONPATH=$PYTHONPATH:$DGL_HOME/python

If you are working on performance critical part, you may want to turn on Cython build:

.. code-block:: bash

   cd python
   python setup.py build_ext --inplace

You could test the build by running the following command and see the path of your local clone.

.. code-block:: bash

   python -c 'import dgl; print(dgl.__path__)'

Unit tests
~~~~~~~~~~

Currently, we use ``nose`` for unit tests.  The organization goes as follows:

* ``backend``: Additional unified tensor interface for supported frameworks.
  The functions there are only used in unit tests, not DGL itself.  Note that
  the code there are not unit tests by themselves.  The additional backend can
  be imported with
  
  .. code-block:: python

     import backend

  The additional backend contains the following files:

  - ``backend/backend_unittest.py``: stub file for all additional tensor
    functions.
  - ``backend/${DGLBACKEND}/__init__.py``: implementations of the stubs
    for the backend ``${DGLBACKEND}``.
  - ``backend/__init__.py``: when imported, it replaces the stub implementations
    with the framework-specific code, depending on the selected backend.  It
    also changes the signature of some existing backend functions to automatically
    select dtypes and contexts.

* ``compute``: All framework-agnostic computation-related unit tests go there.
  Anything inside should not depend on a specific tensor library.  Tensor
  functions not provided in DGL unified tensor interface (i.e. ``dgl.backend``)
  should go into ``backend`` directory.
* ``${DGLBACKEND}`` (e.g. ``pytorch`` and ``mxnet``): All framework-specific
  computation-related unit tests go there.
* ``graph_index``: All unit tests for C++ graph structure implementation go
  there.  The Python API being tested in this directory, if any, should be
  as minimal as possible (usually simple wrappers of corresponding C++
  functions).
* ``lint``: Pylint-related files.
* ``scripts``: Automated test scripts for CI.

To run unit tests, run

.. code-block:: bash

   sh tests/scripts/task_unit_test.sh <your-backend>

where ``<your-backend>`` can be any supported backends (i.e. ``pytorch`` or ``mxnet``).

Contributing Documentations
---------------------------

If the change is about document improvement, we suggest (and strongly suggest if you change the runnable code
there) building the document and render it locally before making a pull request.

Building Docs Locally
`````````````````````

In general building the docs locally involves the following:

1. Install ``sphinx``, ``sphinx-gallery``, and ``sphinx_rtd_theme``.

2. You need both PyTorch and MXNet because our tutorial contains code from both frameworks.  This does *not*
   require knowledge of coding with both frameworks, though.
   
3. Run the following:

   .. code-block:: bash
   
      cd docs
      ./clean.sh
      make html
      cd build/html
      python3 -m http.server 8080
      
4. Open ``http://localhost:8080`` and enjoy your work.

See `here <https://github.com/dmlc/dgl/tree/master/docs>`__ for more details.

Contributing Editorial Changes via GitHub Web Interface
```````````````````````````````````````````````````````

If one is only changing the wording (i.e. not touching the runnable code at all), one can simply do
without the usage of Git CLI:

1. Make your fork by clicking on the **Fork** button in the DGL main repository web page.
2. Make whatever changes in the web interface *within your own fork*.  You can usually tell
   if you are inside your own fork or in the main repository by checking whether you can commit
   to the ``master`` branch: if you cannot, you are in the wrong place.
3. Once done, make a pull request (on the web interface).
4. The committers will review it, suggesting or making changes as necessary.
5. Resolve the suggestions and reviews, and go back to step 4 until approved.
6. Merge it and enjoy your day.

Contributing Code Changes
`````````````````````````

When changing code, please make sure to build it locally and see if it fails.
