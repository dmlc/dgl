import json
import argparse
import os
from .launch_ec2 import launch_ec2, generate_user_data


def run_test(conf_name):
    instances = []
    with open(conf_name, "r") as f:
        conf = json.load(f)
    with open("ec2_script.sh", "r") as f:
        command = f.read()
    for instance_type, tests in conf.items():
        env_list = []
        if "env" in conf[instance_type]:
            for k, v in conf[instance_type]["env"].items():
                os.environ[k] = v
                env_list.append(k)
        instances.append(launch_ec2(generate_user_data(
            command, False, env_list), instance_type, 150))


parser = argparse.ArgumentParser(
    description='run regression test', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--conf", default="../task.json",
                    help="specify task conf path")


args = parser.parse_args()
run_test(args.conf)
