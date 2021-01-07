def set_docker_log():
    import json
    with open("/etc/docker/daemon.json", "r") as f:
        config = json.load(f)
    config["log-driver"] = "awslogs"
    config["log-opts"] = {
        "awslogs-region": "us-west-2",
        "awslogs-group": "/aws/ec2/dgl-ci",
    }
    with open("/etc/docker/daemon.json", "w") as f:
        json.dump(config, f)


set_docker_log()
