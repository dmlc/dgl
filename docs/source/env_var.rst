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

Intel CPU Performance Options
---------------
* ``DGL_CPU_INTEL_KERNEL_ENABLED``:
    * Values: int (default='0')
    * Use dynamic cpu kernels.
    * Suggested values: 1

* ``DGL_CPU_INTEL_KERNEL_LOG``:
    * Values: int (default='0')
    * Show diagnostic message (debug mode).
    * Suggested values: 1
