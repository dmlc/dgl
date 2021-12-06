.. _guide_ko-distributed-tools:

7.4 분산 학습/추론을 런칭하기 위한 툴들
-------------------------------

:ref:`(English Version) <guide-distributed-tools>`

DGL은 분산 학습을 돕는 두 스크립트들을 제공한다.

* *tools/copy_files.py* : 그래프 파티션들을 하나의 그래프로 복사
* *tools/launch.py* : 머신들의 클러스터에서 분산 학습 잡을 시작

*copy_files.py* 는 (그래프가 파티션이 수행된) 한 머신의 파타션된 데이터와 관련 파일들(예, 학습 스크립트)을 (분산 학습이 수행 될) 클러스터에 복사한다. 스크립트는 한 파티션을 해당 파티션을 사용해서 분산 학습 잡이 실행될 머신에 복사한다. 스크립트는 네 개의 인자를 사용한다.

* ``--part_config`` 는 로컬 머신의 파티션된 데이터에 대한 정보를 저장하는 파티션 설정 파일을 지정한다.
* ``--ip_config`` 는 클러스터의 IP 설정 파일을 지정한다.
* ``--workspace`` 는 분산 학습에 관련된 모든 데이터가 저장될 학습 머신의 디렉토리를 지정한다.
* ``--rel_data_path`` 는 파티션된 데이터가 저장될 workspace 디렉토리 아래 상대 경로를 지정한다.
* ``--script_folder`` 는 사용자의 학습 스크립트가 저장될 workspace 디렉토리 아래 상대 경로를 지정한다.

**Note**: *copy_files.py* 는 IP 설정 파일을 기반으로 파티션을 저장할 머신을 찾는다. 따라서, 같은 IP 설정 파일이 *copy_files.py* 과 *launch.py* 에 사용되어야 한다.

DGL은 클러스터에서 분산 학습 잡을 시작하기 위해서 *tools/launch.py* 를 제공한다. 이 스크립트는 다음을 가정한다.

* 파티션된 데이터와 학습 스크립트는 클러스터 또는 클러스터의 모든 머신이 접근 가능한 클로벌 스토리지(예, NFS)로 복사된다.
* (런치 스크립트가 실행되는) 마스터 머신은 다른 모든 머신에 패스워드 없이(passwordless) ssh 접근을 할 수 있다.

**Note**: 런치 스크립트는 클러스터의 머신 중에 하나에서 실행되야 한다.

다음은 클러스터에서 분산 학습 잡을 수행하는 예를 보여준다.

.. code:: none

    python3 tools/launch.py \
    --workspace ~graphsage/ \
    --num_trainers 2 \
    --num_samplers 4 \
    --num_servers 1 \
    --part_config data/ogb-product.json \
    --ip_config ip_config.txt \
    "python3 code/train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 5 --batch-size 1000 --lr 0.1 --num_workers 4"

설정 파일 *ip_config.txt* 은 클러스터의 머신들의 IP 주소들을 저장한다. *ip_config.txt* 의 전형적인 예는 다음과 같다:

.. code:: none

    172.31.19.1
    172.31.23.205
    172.31.29.175
    172.31.16.98

각 줄은 한 머신의 IP 주소이다. 선택적으로 IP 주소 뒤에 트레이너들의 네트워크 통신에 사용될 포트 번호도 지정할 수 있다. 포트 번호가 지정되지 않은 경우 기본 값인 ``30050`` 이 사용된다.

런치 스크립트에서 지정된 workspace는 머신들의 작업 디렉토리로, 학습 스크립트, IP 설정 파일, 파티션 설정 파일 그리고 그래프 파티션들이 저장되는 위치이다. 파일들의 모든 경로들은 workspace의 상대 경로로 지정되어야 한다.

런치 스크립트는 한 머신에서 지정된 수의 학습 잡(``--num_trainers`` )을 생성한다. 또한, 사용자는 각 트레이너에 대한 샘플러 프로세스의 개수(``--num_samplers``)를 정해야 한다. 샘플러 프로세스의 개수는 :func:`~dgl.distributed.initialize` 에서 명시된 worker 프로세스의 개수과 같아야 한다.
