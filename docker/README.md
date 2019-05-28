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

### CPU MXNet image
```bash
docker build -t dgl-mxnet-cpu -f Dockerfile.ci_cpu_mxnet .
```

### GPU MXNet image
```bash
docker build -t dgl-mxnet-gpu -f Dockerfile.ci_gpu_mxnet .
```
