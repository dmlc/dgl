import argparse
import json


parser = argparse.ArgumentParser(description='Change branch in asv conf')
parser.add_argument("--branch", type=str, default="master",
                    help="branch for testing")
args = parser.parse_args()

with open("../asv.conf.json", "r") as f:
    lines = f.readlines()
    for idx in range(len(lines)):
        if lines[idx].strip().startswith('"branches"'):
            print("Change from {}".format(lines[idx]))
            lines[idx] = '    "branches": ["{}"],'.format(args.branch)
            print("To {}".format(lines[idx]))


with open("../asv.conf.json", "w") as f:
    f.writelines(lines)
