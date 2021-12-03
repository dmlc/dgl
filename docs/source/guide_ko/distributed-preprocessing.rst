.. _guide_ko-distributed-preprocessing:

7.1 분산 학습을 위한 전처리
---------------------

:ref:`(English Version) <guide-distributed-preprocessing>`

DGL의 분산 학습을 사용하기 위해서는 그래프 데이터에 대한 전처리가 필요하다. 이 전처리는 두 단계로 구성된다: 1) 그래프를 서브 그래프들로 파티션하기, 2) 노드/에지들에 새로운 ID를 부여하기. 상대적으로 작은 그래프들의 경우, DGL이 제공하는 파티셔닝 API :func:`dgl.distributed.partition_graph` 를 사용해서 위 두 단계를 수행할 수 있다. 이 API는 한 컴퓨터에서 수행된다. 따라서, 그래프가 큰 경우, 이 API를 사용하고 싶다면 큰 컴퓨터를 사용해야 한다. 이 API과 더불어, 여기서는 큰 그래프를 컴퓨터들의 클러스터에서 파티션을 하는 솔루션을 소개한다. (7.1.1 절을 보라)

:func:`dgl.distributed.partition_graph` 는 랜덤 파티션과 `Metis <http://glaros.dtc.umn.edu/gkhome/views/metis>`__ 기반의 파티셔닝을 모두 지원한다. Metis 파티셔닝의 장점은 최소의 에지 컷(edge cut)을 갖는 파티션들을 만들 수 있다는 것이다. 이는 분산 학습 및 추론에서 네트워크 통신을 줄여준다. DGL은 최신 버전의 Metis은 실제(real world)에서 거듭 제곱 법칙의 분포를 갖는 그래프에 최적화되어 있다. 파타셔닝 후, API는 학습시 쉽게 로딩될 수 있는 형태로 파티션된 결과를 만든다.

기본 설정으로 파티션 API는 분산 학습/추론이 실행될 때 노드/에지를 구별하는 것을 돕기 위해서 입력 그래프의 노드와 에지에 새로운 ID를 부여한다. ID를 할당한 후, 파티션 API은 모든 노드 데이터와 에지 데이터를 섞는다. 파티션된 서브 그래프를 생선한 후, 각 서브 그래프는 ``DGLGraph`` 객체로 저장된다. 섞기전의 원본 노드/에지 ID들은 서브 그래프들의 노드/에지 데이터에 `orig_id` 필드에 저장된다. 서브 그래프의 노드 데이터 `dgl.NID` 와 에지 데이터 `dgl.EID` 는 노드/에지들이 reshuffle 후의 전체 그래프의 새로운 노드/에지 ID를 저장한다. 학습이 실행되는 동안, 사용자는 새로운 노드/에지 ID만을 사용한다.

파티션된 결과는 출력 디렉토리의 여러 파일로 저장된다. 이는 한개의 JSON 파일을 포함하는데, 파일 이름은 xxx.json 형태이고, xxx는 파티션 API에 사용된 그래프 이름이다. JSON 파일은 모든 파티션 설정들을 갖는다. 먄약 파티션 API가 새로운 ID를 노드와 에지에 할당하지 않은 경우에는, 추가적으로 두 개의 Numpy 파일; `node_map.npy` 와 `edge_map.npy` 를 생성하는데, 이는 노드/에지 ID와 파티션 ID의 매핑을 저장한다. 만약 그래프에 수십억 개의 노드와 에지가 있다면, 두 파일의 Numpy array는 커질 것인다. 그 이유는 그래프의 각 노드 및 에지에 대해서 하나의 엔트리를 갖기 때문이다. 각 파티션에 대한 폴더는 DGL 포멧으로 파티션 데이터를 저장하는 세 개의 파일이 있다. `graph.dgl` 은 파티션의 그래프 구조와 노드 및 에지에 대한 메타 데이터를 저장하고 있고, `node_feats.dgl` 과 `edge_feats.dlg` 은 파티션에 속하는 노드와 에지의 모든 피쳐들을 저장하고 있다.

.. code-block:: none

    data_root_dir/
        |-- xxx.json                  # partition configuration file in JSON
        |-- node_map.npy              # partition id of each node stored in a numpy array (optional)
        |-- edge_map.npy              # partition id of each edge stored in a numpy array (optional)
        |-- part0/                    # data for partition 0
            |-- node_feats.dgl        # node features stored in binary format
            |-- edge_feats.dgl        # edge features stored in binary format
            |-- graph.dgl             # graph structure of this partition stored in binary format
        |-- part1/
            |-- node_feats.dgl
            |-- edge_feats.dgl
            |-- graph.dgl

로드 밸런싱
~~~~~~~~

그래프를 파티셔닝할 때, Metis의 기본 설정은 각 파티션의 노드 수에 대해서 균형을 맞춘다. 그 결과 주어진 테스크에 따라서 최적이지 않은 구성(suboptimal configuration)이 될 수 있다. 예를 들어, semi-supervised 노드 분류의 경우, 트레이너는 로컬 파티션의 레이블이 있는 노들의 서브셋에 대해서 계산을 수행한다. 그래프의 노드들(레이블이 있는 것과 없는 모든 노드)에 균형을 맞추는 파티셔닝은 계산적인 로드(computational node)가 불균형하게 될 수 있다. 각 파티션에 균형잡힌 워크로드를 얻기 위해서 파티션 API는 각 노드 타입에 대한 노드 수를 고려해서 파티션들에 대한 균형을 만드는 것을 지원한다. 이는 :func:`dgl.distributed.partition_graph` 에서 ``balance_ntypes`` 를 설정하는 것으로 가능하다. 사용자들은 이 기능을 활용해서, 학습 셋, 검증 셋, 그리고 테스트 셋에 다른 노드 타입들이 포함된 것을 고려하게 할 수 있다.

아래 코드는 학습 셋 내에서 그리고 학습 셋 외에 두 가지 노드 타입이 있다는 것을 고려한 코드 예제이다.

.. code:: python

    dgl.distributed.partition_graph(g, 'graph_name', 4, '/tmp/test', balance_ntypes=g.ndata['train_mask'])

노드 타입 균형을 맞추는 것에 더해서, :func:`dgl.distributed.partition_graph` 는 ``balance_edges`` 설정을 통해서 다른 노드 타입들의 노드들의 in-degree들 사이의 균형을 잡는 것을 지원한다. 이는 다른 타입의 노드들에 부속되는 에지들의 개수에 대한 균형을 만든다.

**Note**: :func:`dgl.distributed.partition_graph` 에 전달되는 그래프 이름은 중요한 인자이다. 그 그래프 이름은 :class:`dgl.distributed.DistGraph` 이 분산 그래프를 지정하는데 사용된다. 그래프 이름은 알파벳 문자들과 밑줄 기호만으로 구성되어야 한다.

ID 매핑
~~~~~~

:func:`dgl.distributed.partition_graph` 는 파티셔닝을 하는 과정에서 노드 ID와 에지 ID를 섞고, 노드 데이터와 에지 데이터도 그에 따라서 섞어준다. 학습이 끝나면, 다운스트림 과제를 위해서 계산된 노드 임베딩들을 저장할 필요가 있다. 따라서, 저장된 노드 임베딩을 원본 ID에 따라서 다시 섞어야한다.

`return_mapping=True` 인 경우, :func:`dgl.distributed.partition_graph` 는 섞인 노드/에지 ID와 그것들의 원본 ID 사이의 매핑을 리턴한다. Homogeneous 그래프의 경우, 두 벡터를 리턴한다. 첫번째 벡터는 모든 섞인 노드 ID와 그것의 원본 ID 메핑을, 두번째 벡터는 모든 섞인 에지 ID와 그것의 원본 ID 매핑이다. Heterogeneous 그래프의 경우에는 벡터들의 dictionary 두 개가 리턴된다. 첫번째 dictionary는 각 노드 타입에 대한 매핑을, 두번째 dictionary는 각 에지 타입에 대한 매핑이다.

.. code:: python

    node_map, edge_map = dgl.distributed.partition_graph(g, 'graph_name', 4, '/tmp/test',
                                                         balance_ntypes=g.ndata['train_mask'],
                                                         return_mapping=True)
    # Let's assume that node_emb is saved from the distributed training.
    orig_node_emb = th.zeros(node_emb.shape, dtype=node_emb.dtype)
    orig_node_emb[node_map] = node_emb

7.1.1 분산 파티셔닝
^^^^^^^^^^^^^^^^

큰 그래프를 위해서 DGL은 `ParMetis <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview>`__ 을 사용해서 컴퓨터들의 클러스터에서 그래프를 파티셔닝한다. 이 솔루션은 사용자가 ParMETIS에 맞도록 데이터를 준비하고, ParMETIS에 의해 만들어질 파티션들을 위한 :class:`dgl.DGLGraph` 를 만들기 위해서 DGL 스크립트 `tools/convert_partition.py` 를 사용해야 한다.

**Note**: `convert_partition.py` 는 `pyarrow` 패키지를 사용해서 csv 파일을 로드안다. `pyarrow` 설치하자.

ParMETIS 설치
~~~~~~~~~~~~

ParMETIS는 METIS와 GKLib을 필요로 한다. GKLib 컴파일과 설치는 `here <https://github.com/KarypisLab/GKlib>`__ 에 있는 설명을 참고하자. METIS 컴파일과 설치는 아래 설명을 따라 GIT에서 METIRS를 클론하고 int64 지원을 활성화해서 컴파일한다.

.. code-block:: none

    git clone https://github.com/KarypisLab/METIS.git
    make config shared=1 cc=gcc prefix=~/local i64=1
    make install

여기서부터는 PartMETIS를 직접 컴파일하고 설치하는 것이 필요하다. 아래 명령을 사용해서 ParMETIS의 DGL 브랜치를 클론한다.

.. code-block:: none

    git clone --branch dgl https://github.com/KarypisLab/ParMETIS.git

그리고, ParMETIS를 컴파일하고 설치한다.

.. code-block:: none

    make config cc=mpicc prefix=~/local
    make install

ParMETIS를 실행하기 전에, 두 환경 변수들, `PATH`와 `LD_LIBRARY_PATH`을 설정해야 한다: 

.. code-block:: none

    export PATH=$PATH:$HOME/local/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/local/lib/

ParMETIS를 위한 입력 포멧
~~~~~~~~~~~~~~~~~~~~~

ParMETIS의 입력 그래프는 다음 이름들을 사용해서 세 개의 파일들에 저장된다: `xxx_nodes.txt` , `xxx_edges.txt` 와 `xxx_stats.txt`. 여기서 `xxx` 는 그래프 이름이다.

`xxx_nodes.txt` 의 각 행은 다음 형식으로 노드에 대한 정보를 담고 있다.

.. code-block:: none

    <node_type> <weight1> ... <orig_type_node_id> <attributes>

모든 필드들은 공백 문자로 구분된다.

* `<node_type>` 은 정수 값이다. Homogeneous 그래프에서는 항상 0이고, heterogenous 그래프에서는 그 값이 각 노드의 타입을 의미한다.
* `<weight1>`, `<weight2>`, 등은 정수 값들인데, ParMETIS가 그래프 파티션들의 균형을 맞출 때 노드 가중치로 사용하는 값들이다. 사용자가 노드 가중치를 명시하지 않는 경우, ParMETIS는 각 파티션의 노드 수에 대한 균형을 고려해서 파티션을 나눈다 (좋은 학습 속도를 얻기 위해서는 그래프 파티션들의 균헝을 맞추는 것이 중요하다). 하지만, 이 기본 전략은 많은 use case들에 충분하지 않을 수 있다. 예를 들어, heterogeneous 그래프의 경우, 우리는 모든 파티션들이 각 노드 타입별로 비슷한 개수의 노드들을 갖도록 그래프에 대한 파티션을 나누고 싶다. 아래 토이 예제는 노드 가중치를 사용해서 다른 테입들의 노드 개수의 균형을 맞추것을 어떻게 하는지 보여준다.
* `<orig_type_node_id>` 은 노드 타입에서의 노드 ID를 표현하는 정수 값이다. DGL에서 각 타입의 노드들은 0부터 시작하는 ID가 부여된다. Homogeneous 그래프에서 이 필드는 노드 ID의 값도 동일하다.
* `<attributes>` 는 선택적인 필드들이다. 이는 임의의 값을 저장하는데 사용될 수 있으며, ParMETIS는 이 필드들을 사용하지 않는다. 잠재적으로는 homogenous 그래프들의 경우 노드 피쳐들과 에지 피쳐들을 이 필드에 저장할 수 있다.
* 행(row) ID는 그래프의 *homogeneous* ID를 의미한다 (모든 노드에 고유한 ID가 할당된다). 같은 타입의 모든 노드들에 ID는 연속된 값으로 부여된다. 즉, 같은 타입의 노드들은 `xxx_notes.txt` 파일에 함께 저장되어야 한다.

다음은 두 노드 타입을 갖는 heterogenous 그래프의 노트 파일 예이다. 노드 타입 0은 세 개의 노드를 갖고 있고, 노드 타입 1은 네 개의 노드들을 갖는다. 두 노드 가중치를 사용해서 ParMETIS느 노드 타입 0에 속한 노드 개수와 노드 타입 1에 속한 노드 개수가 대략 같도록 파티션 나눈다.

.. code-block:: none

    0 1 0 0
    0 1 0 1
    0 1 0 2
    1 0 1 0
    1 0 1 1
    1 0 1 2
    1 0 1 3

비슷하게, `xxx_edges.txt` 의 각 행은 아래 형식으로 에지에 대한 정보를 저장한다.

.. code-block:: none

    <src_id> <dst_id> <type_edge_id> <edge_type> <attributes>

모든 필드들은 공백 문자로 구분된다.

* `<src_id>` 는 소스 노드의 *homogeneous* ID이다.
* `<dst_id>` 는 목적지 노드의 *homogeneous* ID이다.
* `<type_edge_id>` 는 에지 타입에 대한 에지 ID이다.
* `<attributes>` 는 선택적인 필드들이다. 임의의 값을 저장하는데 사용할 수 있는데, ParMETIS는 이 필드를 사용하지 않는다.

**Note**: 에지 파일에 중복된 에지나 셀프-룹을 갖는 에지가 없어야 한다.

`xxx_stats.txt` 는 그래프에 대한 기본적인 통계들을 저장한다. 이 파일은 공백으로 구분되는 세 필드들로 구성된 단 한 줄만 갖는다.

.. code-block:: none

    <num_nodes> <num_edges> <num_node_weights>

* `num_nodes` 는 노드 타입을 상관하지 않고 전체 노드 수를 저장한다.
* `num_edges` 는 에지 타입을 상관하지 않고 전체 에지 수를 저장한다.
* `num_node_weights` 는 노드 파일의 노드 가중치 수를 저장한다.

ParMETIS 실행하기 및 결과 포멧들
~~~~~~~~~~~~~~~~~~~~~~~~~~

ParMETIS는 `pm_dglpart` 명령이 실행된 머신에서 세 파일들에 저장된 그래프를 로드하고, 클러스터의 모든 머신에 데이터를 분산하고, ParMETIS를 실행해서 그래프의 파티션을 나누는 명령 `pm_dglpart` 을 포함하고 있다. 이 명령의 수행이 완료되면, 각 파타션에 대해서 세 개의 파일이 생성된다: `p<part_id>-xxx_nodes.txt`, `p<part_id>-xxx_edges.txt`, `p<part_id>-xxx_stats.txt`

**Note**: ParMETIS는 파티셔닝을 수행하면서 노드들에 ID를 재할당한다. ID 재할당이 끝나면, 한 파티션의 노드들은 연속된 ID값을 갖는다; 더 나아가, 같은 타입의 노드들은 연속된 ID들을 부여 받는다.

`p<part_id>-xxx_nodes.txt` 는 파티션의 노드 데이터를 저장한다. 각 행은 한 노드에 대한 다음 정보들을 담고 있다.

.. code-block:: none

    <node_id> <node_type> <weight1> ... <orig_type_node_id> <attributes>

* `<node_id>` 는 ID 재할당 후의 *homogeneous* 노드 ID이다.
* `<node_type>` 는 노드 타입이다.
* `<weight1>` 는 ParMETIS가 사용하는 노드 가중치이다.
* `<orig_type_node_id>` 는 입력 heterogeneous 그래프의 특정 노드 티입에 대한 원본 노드 ID이다.
* `<attributes>` 는 선택적인 필드들로 입력 노드 파일에서 임의의 값을 갖는다.

`p<part_id>-xxx_edges.txt` 는 파티션의 에지 데이터를 저장한다. 각 행은 한 에지에 대한 다음 정보를 담고 있다.

.. code-block:: none

    <src_id> <dst_id> <orig_src_id> <orig_dst_id> <orig_type_edge_id> <edge_type> <attributes>

* `<src_id>` 는 ID 재할당 후의 소스 노드의 *homogeneous* ID이다.
* `<dst_id>` 는 ID 재할당 후의 목적지 노드의 *homogeneous* ID이다.
* `<orig_src_id>` 는 입력 그래프의 소스 노드에 대한 *homogeneous* ID이다.
* `<orig_dst_id>` 는 입력 그래프의 목적지 노드에 대한 *homogeneous* ID이다.
* `<orig_type_edge_id>` 는 입력 그래프의 특정 에지 타입에 대한 에지 ID이다.
* `<edge_type>` 은 에지 타입이다.
* `<attributes>` 는 선택적인 필드들로 입력 에지 파일에서 임의의 에지 속성 값을 갖는다.

`pm_dglpart` 이 실행된 때, 세 입력 파일들(`xxx_nodes.txt`, `xxx_edges.txt`, `xxx_stats.txt`)은 `pm_dglpart` 명령이 실행된 디렉토리와 같은 곳에 있어야 한다. 다음 명령은 네 개의 ParMETIS 프로세스를 실행해서, `xxx` 라는 이름의 그래프를 8개의 파티션으로 나눈다 (각 프로세스는 2개의 파티션을 담당한다).

.. code-block:: none

    mpirun -np 4 pm_dglpart xxx 2

ParMETIS 결과들을 DGLGraph로 변환하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DGL은 `convert_partition.py` 라는 스크립트를 제공한다. 이는 `tool` 디렉토리에 있는데, 파티션 파일들에 있는 데이터를 :class:`dgl.DGLGraph` 객체로 변환하고 파일들에 저장하는 역할을 한다. **Note** `convert_partition.py` 는 단일 머신에서 실행된다. 향후, 우리는 이를 확장해서 여러 머신들에 걸쳐서 데이터를 병렬로 변환하도록 만들 것이다.  **Note**: csv 파일로 저장된 데이터를 로딩하기 위해서 `pyarrow` 패키지를 설치하자.

`convert_partition.py` 는 다음 인자들을 받는다:

* `--input-dir INPUT_DIR` 는 ParMETIS가 생성한 파티션 파일들이 있는 디렉토리를 지정한다.
* `--graph-name GRAPH_NAME` 는 그래프 이름을 지정한다.
* `--schema SCHEMA` 는 입력 heterogeneous 그래프의 스키마를 명시하는 파일이다. 스키마 파일은 JSON 파일로서, 노드 타입들과 에지 타입들을 나열하고, 또한 각 노드 타입 및 에지 타입에 대한 homogeneous ID의 범위를 포함한다.
* `--num-parts NUM_PARTS` 는 파티션의 개수를 명시한다.
* `--num-node-weights NUM_NODE_WEIGHTS` 는 ParMETIS가 파티션들의 균형을 위해서 사용한 노드 가중치의 개수를 지정한다.
* `[--workspace WORKSPACE]` 는 선택적인 인자로, 중간 결과들을 저장할 workspace 디렉토리를 지정한다.
* `[--node-attr-dtype NODE_ATTR_DTYPE]` 는 선택적인 인자로, 노드 파일들의 나머지 필드인 `<attributes>` 에 저장된 노드 속성들의 데이터 타입을 명시한다.
* `[--edge-attr-dtype EDGE_ATTR_DTYPE]` 는 선택적인 인자로, 에지 파일들의 나머지 필드인 `<attributes>` 에 저장된 에지 속성들의 데이터 타입을 명시한다.
* `--output OUTPUT` 는 파티션 결과들이 저장될 출력 디렉토리를 지정한다.

`convert_partition.py` 의 결과 파일들은 다음과 같다:

.. code-block:: none

    data_root_dir/
        |-- xxx.json                  # partition configuration file in JSON
        |-- part0/                    # data for partition 0
            |-- node_feats.dgl        # node features stored in binary format (optional)
            |-- edge_feats.dgl        # edge features stored in binary format (optional)
            |-- graph.dgl             # graph structure of this partition stored in binary format
        |-- part1/
            |-- node_feats.dgl
            |-- edge_feats.dgl
            |-- graph.dgl

**Note**: 노드 속성 또는 에지 속성의 데이터 타입이 명시된다면, `convert_partition.py` 는 모든 타입의 모든 노드들 및 에지들이 꼭 이 속성들을 갖는다고 가정한다. 따라서, 다른 타입의 노드들이나 에지들이 서로 다른 개수의 속성을 갖는다면, 사용자는 이를 직접 만들어야 한다.

다음은 `convert_partition.py` 를 위한 OGBN-MAG의 스키마 예제이다. 이는 두 필드를 갖는다: `nid` 와 `eid`. `nid` 안에는, 모든 노드 타입들이 나열되어 있고, 각 노드 타입에 대한 homogeneous ID 범위도 포함되어 있다; `eid` 안에는, 모든 에지 타입들이 나열되어 있고, 각 에지 타입에 대한 homogeneous ID 범위도 포함되어 있다.

.. code-block:: none

    {
    "nid": {
        "author": [
            0,
            1134649
        ],
        "field_of_study": [
            1134649,
            1194614
        ],
        "institution": [
            1194614,
            1203354
        ],
        "paper": [
            1203354,
            1939743
        ]
    },
    "eid": {
        "affiliated_with": [
            0,
            1043998
        ],
        "writes": [
            1043998,
            8189658
        ],
        "rev-has_topic": [
            8189658,
            15694736
        ],
        "rev-affiliated_with": [
            15694736,
            16738734
        ],
        "cites": [
            16738734,
            22155005
        ],
        "has_topic": [
            22155005,
            29660083
        ],
        "rev-cites": [
            29660083,
            35076354
        ],
        "rev-writes": [
            35076354,
            42222014
        ]
    }
    }

아래 코드는 스키마 파일을 만드는 예제이다.

.. code-block:: none

    nid_ranges = {}
    eid_ranges = {}
    for ntype in hg.ntypes:
        ntype_id = hg.get_ntype_id(ntype)
        nid = th.nonzero(g.ndata[dgl.NTYPE] == ntype_id, as_tuple=True)[0]
        nid_ranges[ntype] = [int(nid[0]), int(nid[-1] + 1)]

    for etype in hg.etypes:
        etype_id = hg.get_etype_id(etype)
        eid = th.nonzero(g.edata[dgl.ETYPE] == etype_id, as_tuple=True)[0]
        eid_ranges[etype] = [int(eid[0]), int(eid[-1] + 1)]
    with open('mag.json', 'w') as outfile:
        json.dump({'nid': nid_ranges, 'eid': eid_ranges}, outfile, indent=4)

Heterogeneous 그래프에 대한 노드/에지 피처들 생성하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`convert_partition.py` 이 만든 :class:`dgl.DGLGraph` 아웃풋은 heterogeneous 그래프 파티션들을 homogeneous 그래프로 저장한다. 노드 데이터는 `orig_id` 라는 필드를 갖는데, 이는 원본 heterogeneous 그래프의 특정 노드 타입의 노드 ID들을 저장하고, `NTYPE` 의 필드는 노드 타입을 저장한다. 추가로, 이는 `inner_node` 라는 노드 데이터를 저장하는데, 이는 그래프 파티션의 노드가 파티션이 할당되어 있는지 여부를 알려준다. 만약 어떤 노드가 파티션에 할당되었다면, `inner_node` 는 1을 갖고, 반대의 경우에는 0을 갖는다. **Note**: 그래프 파티션은 몇 개의 HALO 노드들을 포함하는데, 이는 다른 파티션에 할당된 것지만, 이 그래프 파티션의 몇 개의 에지와 연결되어 있는 것들이다. 이 정보를 사용해서, 우리는 별도로 각 노드 타입에 대한 노드 피쳐들을 구성할 수 있으며, 이들을 `<node_type>/<feature_name>` 를 키로 갖고 값은 노드 피쳐 벡터인 dictionary에 저장할 수 있다. 아래 코드는 노드 피쳐 dictionary를 구성하는 방법을 보여준다. 텐서들의 dictionary가 만들어지면, 이는 파일에 저장된다.

.. code-block:: none

    node_data = {}
    for ntype in hg.ntypes:
        local_node_idx = th.logical_and(part.ndata['inner_node'].bool(),
                                        part.ndata[dgl.NTYPE] == hg.get_ntype_id(ntype))
        local_nodes = part.ndata['orig_id'][local_node_idx].numpy()
        for name in hg.nodes[ntype].data:
            node_data[ntype + '/' + name] = hg.nodes[ntype].data[name][local_nodes]
    dgl.data.utils.save_tensors(metadata['part-{}'.format(part_id)]['node_feats'], node_data)

에지 피쳐도 비슷한 방법으로 구성할 수 있다. 차이점은 :class:`dgl.DGLGraph` 의 모든 에지들이 파티션에 포함된다는 점이다. 그래서, 구성 방법은 더 간단하다.

.. code-block:: none

    edge_data = {}
    for etype in hg.etypes:
        local_edges = subg.edata['orig_id'][subg.edata[dgl.ETYPE] == hg.get_etype_id(etype)]
        for name in hg.edges[etype].data:
            edge_data[etype + '/' + name] = hg.edges[etype].data[name][local_edges]
    dgl.data.utils.save_tensors(metadata['part-{}'.format(part_id)]['edge_feats'], edge_data)
