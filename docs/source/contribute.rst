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

The python code style configure file is ``tests/scripts/pylintrc``. We tweak it a little bit from
the standard. For example, following variable names are accepted:

* ``i,j,k``: for loop variables
* ``u,v``: for representing nodes
* ``e``: for representing edges
* ``g``: for representing graph
* ``fn``: for representing functions
* ``n,m``: for representing sizes
* ``w,x,y``: for representing weight, input, output tensors
* ``_``: for unused variables

Building and Testing
--------------------

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
``````````

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

Building documents
------------------

If the change is about document improvement, we suggest build the document and render it locally
before pull request. See instructions `here <https://github.com/dmlc/dgl/tree/master/docs>`__.

Data hosting
------------

If the change is about new models or applications, it is very common to have some data files. Data
files are not allowed to be uploaded to our repository. Instead, they should be hosted on the
cloud storage service (e.g. dropbox, Amazon S3) and downloaded on-the-fly. See our :ref:`dataset APIs <apidata>`
for more details. All the dataset of current DGL models are hosted on Amazon S3. If you want your
dataset to be hosted as well, please post in our `discussion forum <https://discuss.dgl.ai>`__.
