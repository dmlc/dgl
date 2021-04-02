import argparse
import json
import os


def change_branch(branch_str: str):
    branches = branch_str.split(",")
    branch = branches[0]
    with_branch = None
    if len(branches) == 2:
        with_branch = branches[1]

    if not branch.startswith("origin/"):
        branch = "origin/" + branch
    with open("../asv.conf.json", "r") as f:
        lines = f.readlines()
        for idx in range(len(lines)):
            if lines[idx].strip().startswith('"branches"'):
                print("Change from {}".format(lines[idx]))
                if with_branch is None:
                    lines[idx] = '    "branches": ["{}"],'.format(branch)
                else:
                    lines[idx] = '    "branches": ["{}", "{}"],'.format(
                        branch, with_branch)
                print("To {}".format(lines[idx]))

    with open("../asv.conf.json", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    if "BRANCH_STR" in os.environ:
        change_branch(os.environ["BRANCH_STR"])
