.. _guide_ko-data-pipeline-download:

4.2 Raw 데이터 다운로드하기 (optional)
---------------------------------

:ref:`(English Version) <guide-data-pipeline-download>`

로컬 디스크에 데이터셋이 이미 존재한다면, ``raw_dir`` 디렉토리에 있어야 한다. 만약 데이터를 다운로드하고 특정 디렉토리에 옮기는 일을 직접 수행하지 않고 코드를 실행하고 어디서나 실행하고 싶다면, ``download()`` 구현해서 이를 자동화할 수 있다.

데이터셋이 zip 파일 포멧인 경우, zip 파일 추출을 자동을 해주는 :class:`dgl.data.DGLBuiltinDataset` 클래스를 상속해서 ``MyDataset`` 클래스를 만들자. 그렇지 않은 경우 :class:`~dgl.data.QM7bDataset` 처럼 ``download()`` 함수를 직접 구현한다:

.. code:: 

    import os
    from dgl.data.utils import download
    
    def download(self):
        # path to store the file
        file_path = os.path.join(self.raw_dir, self.name + '.mat')
        # download file
        download(self.url, path=file_path)

위 코드는 .mat 파일을 ``self.raw_dir`` 디렉토리에 다운로드한다. 만약 파일 포멧이 .gz, .tar, .tar.gz 또는 .tgz 이라면, :func:`~dgl.data.utils.extract_archive` 함수로 파일들을 추출하자. 다음 코드는 :class:`~dgl.data.BitcoinOTCDataset` 에서 .gz 파일을 다운로드하는 예이다:

.. code:: 

    from dgl.data.utils import download, check_sha1
    
    def download(self):
        # path to store the file
        # make sure to use the same suffix as the original file name's
        gz_file_path = os.path.join(self.raw_dir, self.name + '.csv.gz')
        # download file
        download(self.url, path=gz_file_path)
        # check SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name + '.csv.gz'))
        # extract file to directory `self.name` under `self.raw_dir`
        self._extract_gz(gz_file_path, self.raw_path)

위 코드는 ``self.raw_dir`` 디렉토리 아래의 ``self.name`` 서브 디렉토리에 파일을 추출한다. 만약 zip 파일을 다루기 위해서 :class:`dgl.data.DGLBuiltinDataset` 를 상속해서 사용했다면, 파일들은 자동으로 ``self.name`` 디렉토리로 추출될 것이다.

추가적으로, 다운로드한 파일에 대한 SHA-1 값 검증을 수행해서 파일이 변경되었는지 확인하는 것도 위 예제처럼 구현할 수 있다.