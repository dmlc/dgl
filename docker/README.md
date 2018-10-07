## Build docker image for CI
### CPU image
docker build -t dgl_cpu -f Dockerfile.ci_cpu .

### GPU image
docker build -t dgl_gpu -f Dockerfile.ci_gpu .
