## Build docker image for CI

### CPU image
docker build -t dgl-cpu -f Dockerfile.ci_cpu .

### CPU MXNet image
docker build -t dgl-mxnet-cpu -f Dockerfile.ci_cpu_mxnet .

### GPU image
docker build -t dgl-gpu -f Dockerfile.ci_gpu .

### GPU MXNet image
sudo docker build -t dgl-mxnet-gpu -f Dockerfile.ci_gpu_mxnet .

### Lint image
docker build -t dgl-lint -f Dockerfile.ci_lint .
