from setuptools import setup, find_packages


setup(
    name="dgi",
    version='0.0.1',
    author="Peiqi Yin",
    author_email="yinpeiqi@amazon.com",
    description="Deep Graph Inference (DGI): mini batch inference helper",
    packages=find_packages(),
    install_requires=[
        'dgl>=0.8.0',
        'numpy>=1.14.0',
        'torch',
        'tqdm',
        'pynvml',
        'ogb',
    ],
    url='https://github.com/dmlc/dgl',
)
