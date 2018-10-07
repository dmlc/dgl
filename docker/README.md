## Build docker image for CI

### CPU image
docker build -t dgl-cpu -f Dockerfile.ci_cpu .

### GPU image
docker build -t dgl-gpu -f Dockerfile.ci_gpu .
