## Build docker image for CI

### CPU image
```bash
docker build -t dgl-cpu -f Dockerfile.ci_cpu .
```

### GPU image
```bash
docker build -t dgl-gpu -f Dockerfile.ci_gpu .
```

### Lint image
```bash
docker build -t dgl-lint -f Dockerfile.ci_lint .
```

### CPU image for kg
```bash
wget https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/FB15k.zip -P install/
docker build -t dgl-cpu:torch-1.2.0 -f Dockerfile.ci_cpu_torch_1.2.0 .
```

### GPU image for kg
```bash
wget https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/FB15k.zip -P install/
docker build -t dgl-gpu:torch-1.2.0 -f Dockerfile.ci_gpu_torch_1.2.0 .
```
