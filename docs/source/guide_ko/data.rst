.. _guide_ko-data-pipeline:

4장: 그래프 데이터 파이프라인
======================

:ref:`(English Version) <guide-data-pipeline>`

DGL은 :ref:`apidata` 에서 일반적으로 많이 사용되는 그래프 데이터셋을 구현하고 있다. 이것들은 :class:`dgl.data.DGLDataset` 클래스에서 정의하고 있는 표준 파이프라인을 따른다. DGL은 :class:`dgl.data.DGLDataset` 의 서브클래스로 그래프 데이터 프로세싱하는 것을 강하게 권장한다. 이는 파이프라인이 그래프 데이터를 로딩하고, 처리하고, 저장하는데 대한 간단하고 깔끔한 방법을 제공하기 때문이다.

로드맵
----

이 장은 커스텀 DGL-Dataset를 만드는 방법을 소개한다. 이를 위해 다음 절들에서 파이프라인이 어떻게 동작하는지 설명하고, 각 파이프라인의 컴포넌트를 구현하는 방법을 보여준다.

* :ref:`guide_ko-data-pipeline-dataset`
* :ref:`guide_ko-data-pipeline-download`
* :ref:`guide_ko-data-pipeline-process`
* :ref:`guide_ko-data-pipeline-savenload`
* :ref:`guide_ko-data-pipeline-loadogb`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    data-dataset
    data-download
    data-process
    data-savenload
    data-loadogb