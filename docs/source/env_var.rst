Environment Variables
=====================

Backend Options
---------------
* ``DGLBACKEND``:
    * Values: String (default='pytorch')
    * The backend deep learning framework for DGL.
    * Choices:
        * 'pytorch': use PyTorch as the backend implementation.        
        * 'tensorflow': use Apache TensorFlow as the backend implementation.
        * 'mxnet': use Apache MXNet as the backend implementation.

Data Repository
---------------
* ``DGL_REPO``:
    * Values: String (default='https://data.dgl.ai/')
    * The repository url to be used for DGL datasets and pre-trained models.
    * Suggested values:
        * 'https://data.dgl.ai/': DGL repo for Global Region.
        * 'https://dgl-data.s3.cn-north-1.amazonaws.com.cn/': DGL repo for Mainland China
* ``DGL_DOWNLOAD_DIR``:
    * Values: String (default="${HOME}/.dgl")
    * The local directory to cache the downloaded data.

GPU Options
-----------
* ``DGL_USE_CUDA_MEMORY_POOL``:
    * values: `true`/`1` to enable, or `false`/`0` to disable (default=`false`).
    * If enabled, GPU allocations will be saved to pool, to be re-used for
    * subsequent allocatoins. This may enable faster execution at the cost
      of higher memory consumption.
