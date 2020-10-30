.. _guide_cn-data-pipeline-download:

4.2 下载原始数据（可选）
--------------------------------

:ref:`(English Version) <guide-data-pipeline-download>`

If a dataset is already in local disk, make sure it’s in directory
``raw_dir``. If one wants to run the code anywhere without bothering to
download and move data to the right directory, one can do it
automatically by implementing function ``download()``.

如果用户的数据集已经在本地磁盘中，请确保它在目录 ``raw_dir`` 中。
如果用户想在任何地方运行代码而又不必费心下载数据并将其移动到正确的目录中，则可以通过实现函数 ``download()`` 来自动完成。

If the dataset is a zip file, make ``MyDataset`` inherit from
:class:`dgl.data.DGLBuiltinDataset` class, which handles the zip file extraction for us. Otherwise,
one needs to implement ``download()`` like in :class:`~dgl.data.QM7bDataset`:

如果数据集是一个zip文件，请使 ``MyDataset`` 继承 :class:`dgl.data.DGLBuiltinDataset` 类。该类处理了zip文件的解压缩。
否则，请像 :class:`~dgl.data.QM7bDataset` 里一样实现 ``download()`` ：

.. code:: 

    import os
    from dgl.data.utils import download
    
    def download(self):
        # path to store the file
        # 存储文件的路径
        file_path = os.path.join(self.raw_dir, self.name + '.mat')
        # download file
        # 下载文件
        download(self.url, path=file_path)

The above code downloads a .mat file to directory ``self.raw_dir``. If
the file is a .gz, .tar, .tar.gz or .tgz file, use :func:`~dgl.data.utils.extract_archive`
function to extract. The following code shows how to download a .gz file
in :class:`~dgl.data.BitcoinOTCDataset`:

上面的代码将一个.mat文件下载到目录 ``self.raw_dir``。如果文件是.gz、.tar、.tar.gz或.tgz文件，请使用
:func:`~dgl.data.utils.extract_archive` 函数进行解压缩。以下代码展示了如何在
:class:`~dgl.data.BitcoinOTCDataset` 类中下载一个.gz文件：

.. code:: 

    from dgl.data.utils import download, check_sha1
    
    def download(self):
        # path to store the file
        # make sure to use the same suffix as the original file name's
        # 存储文件的路径，请确保使用与原始文件名相同的后缀
        gz_file_path = os.path.join(self.raw_dir, self.name + '.csv.gz')
        # download file
        # 下载文件
        download(self.url, path=gz_file_path)
        # check SHA-1
        # 检查 SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name + '.csv.gz'))
        # extract file to directory `self.name` under `self.raw_dir`
        # 将文件解压缩到目录self.raw_dir下的self.name目录中
        self._extract_gz(gz_file_path, self.raw_path)

The above code will extract the file into directory ``self.name`` under
``self.raw_dir``. If the class inherits from :class:`dgl.data.DGLBuiltinDataset`
to handle zip file, it will extract the file into directory ``self.name`` 
as well.

上面的代码会将文件解压缩到 ``self.raw_dir`` 下的目录 ``self.name`` 中。
如果该类继承自 :class:`dgl.data.DGLBuiltinDataset` 来处理zip文件，
则它也会将文件解压缩到目录 ``self.name`` 中。

Optionally, one can check SHA-1 string of the downloaded file as the
example above does, in case the author changed the file in the remote
server some day.

一个可选项是按照上面的示例检查下载后文件的SHA-1字符串，以防作者在远程服务器上更改了文件。