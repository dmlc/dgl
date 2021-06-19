## Training Scripts for distributed training

1. Partition data

Partition FB15k:

```bash
./partition.sh  ../data FB15k 4
```

Partition Freebase:

```bash
./partition.sh  ../data Freebase 4
```

2. Modify `ip_config.txt` and copy dgl-ke to all the machines

3. Run

```bash
./launch.sh \
  ~/dgl/apps/kg/distributed \
  ./fb15k_transe_l2.sh \
  ubuntu ~/mykey.pem
```