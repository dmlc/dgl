Regression Test Suite
========================

### Spec of task.json
```json
# Note the test will be run if the name specified below is a substring of the full test name.
# The fullname of "benchmarks/model_acc/bench_sage_ns.track_acc" will be "model_acc.bench_sage_ns.track_acc". Test will be run if it contains any keyword.
# For example, "model_acc" will run all the tests under "model_acc" folder
# "bench_sage" will run both "bench_sage" and "bench_sage_ns"
# "bench_sage." will only run "bench_sage"
# "ns" will run any tests name contains "ms"
# "" will run all tests
{
    "c5.9xlarge": { # The instance type to run the test
        "tests": [
            "bench_sage" # The test to be run on this instance
        ],
        "env": {
            "DEVICE": "cpu" # The environment variable passed to publish.sh
        }
    },
    "g4dn.2xlarge": {
        ...
    }
}
```


### Environment variable
- `MOUNT_PATH` specify the directory in the host to be mapped into docker, if exists will map the `MOUNT_PATH`(in host) to `/tmp/dataset`(in docker)
- `INSTANCE_TYPE` specify the current instance type
- `DGL_REG_CONF` specify the path to `task.json`, which is relative to the repo root. If specified, must specify `INSTANCE_TYPE` also