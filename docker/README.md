## Build docker image for CI

### CPU image
docker build -t dgl-cpu -f Dockerfile.ci_cpu .

### GPU image
docker build -t dgl-gpu -f Dockerfile.ci_gpu .

### Lint image
docker build -t dgl-lint -f Dockerfile.ci_lint .

### CPU MXNet image
docker build -t dgl-mxnet-cpu -f Dockerfile.ci_cpu_mxnet .

### GPU MXNet image
docker build -t dgl-mxnet-gpu -f Dockerfile.ci_gpu_mxnet .
