Regression Test Suite
========================

## Launch the whole regression test
This will read the conf from the `benchmarks/task.json`. 
```bash
python run_reg_test.py
```

### Spec of task.json
```json
# Note the test will be run if the name specified below is a substring of the full test name.
# For example, "bench_sage_ns.track_acc/bench_sage_ns.track_time" and "bench_sage_ns.track_acc/bench_sage_ns.track_time" will both qualified if filtered with "bench_sage". If you want to exclude "bench_sage_ns", you can use "bench_sage." as the filter keyword.
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

## Details
The launch script will do the following things:
- install aws cloudwatch agent to record the running output
- mount the dataset EFS to /mnt/efs/fs1
- pass the specified environment variable(GIT_URL, GIT_BRANCH) on the current machine to the remote machine
- 