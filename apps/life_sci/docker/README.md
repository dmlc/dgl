# Build Docker Image for CI

Docker images are used by the CI and release script. Make sure to install necessary requirements in it.

## To build

```bash
docker build -t dgllib/dgllife-ci-cpu:latest -f Dockerfile.ci_cpu .
```

```bash
docker build -t dgllib/dgllife-ci-gpu:latest -f Dockerfile.ci_gpu .
```

## To push

```bash
docker push dgllib/dgllife-ci-cpu:latest
```

```bash
docker push dgllib/dgllife-ci-gpu:latest
```
