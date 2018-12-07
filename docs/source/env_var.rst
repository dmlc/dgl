Environment Variables
=====================

Backend Options
---------------
* ``DGLBACKEND``:
    * Values: String (default='pytorch')
    * The backend deep lerarning framework for DGL.
    * Choices:
        * 'pytorch': use PyTorch as the backend implentation.
        * 'mxnet': use Apache MXNet as the backend implementation.

Data Repository
---------------
* ``DGL_REPO``:
    * Values: String (default='https://s3.us-east-2.amazonaws.com/dgl.ai/')
    * The repository url to be used for DGL datasets and pre-trained models.
    * Suggested values:
        * 'https://s3.us-east-2.amazonaws.com/dgl.ai/': DGL repo for U.S.
        * 'https://s3-ap-southeast-1.amazonaws.com/dgl.ai.asia/': DGL repo for Asia
* ``DGL_DOWNLOAD_DIR``:
    * Values: String (default="${HOME}/.dgl")
    * The local directory to cache the downloaded data.
