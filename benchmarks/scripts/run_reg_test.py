import json
import argparse
from .launch_ec2 import launch_ec2, generate_user_data


def run_test(conf_name):
    instances = []
    with open(conf_name, "r") as f:
        conf = json.load(f)
    with open("ec2_script.sh", "r") as f:
            command = f.read()
    for instance_type, tests in conf.items():
        instances.append(launch_ec2(generate_user_data(
            command, False, []), instance_type, 150))


parser = argparse.ArgumentParser(
    description='run regression test', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--conf", default="../task.json",
                    help="specify task conf path")


args = parser.parse_args()
run_test(args.conf)
