.. _guide_ko-nn:

3장: GNN 모듈 만들기
=================

:ref:`(English Version) <guide-nn>`

DGL NN 모듈은 GNN 모델을 만드는데 필요한 빌딩 블록들로 구성되어 있다. NN 모듈은 백엔드로 사용되는 DNN 프레임워크에 따라 `Pytorch’s NN Module <https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/module.html>`__ , `MXNet Gluon’s NN Block  <http://mxnet.incubator.apache.org/versions/1.6/api/python/docs/api/gluon/nn/index.html>`__ 그리고 `TensorFlow’s Keras Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`__ 를 상속한다. DGL NN 모듈에서, 생성 함수에서의 파라메터 등록과 forward 함수에서 텐서 연산은 백엔드 프레임워크의 것과 동일하다. 이런 방식의 구현덕에 DGL 코드는 백엔드 프레임워크 코드와 원활하게 통합될 수 있다. 주요 차이점은 DGL 고유의 메시지 전달 연산에 존재한다.

DGL은 일반적으로 많이 사용되는 :ref:`apinn-pytorch-conv` , :ref:`apinn-pytorch-dense-conv` , :ref:`apinn-pytorch-pooling` 와 :ref:`apinn-pytorch-util` 를 포함하고 있고. 여러분의 기여를 환영한다.

이 장에서는 PyTorch 백엔드를 사용한 :class:`~dgl.nn.pytorch.conv.SAGEConv` 를 예제로 커스텀 DGL NN 모듈을 만드는 방법을 소개한다.

로드맵
----

* :ref:`guide_ko-nn-construction`
* :ref:`guide_ko-nn-forward`
* :ref:`guide_ko-nn-heterograph`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    nn-construction
    nn-forward
    nn-heterograph
