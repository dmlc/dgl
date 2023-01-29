import enum
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pytest
import requests


class JobStatus(enum.Enum):
    SUCCESS = 0
    FAIL = 1
    SKIP = 2


JENKINS_STATUS_MAPPING = {
    "SUCCESS": JobStatus.SUCCESS,
    "ABORTED": JobStatus.FAIL,
    "FAILED": JobStatus.FAIL,
    "IN_PROGRESS": JobStatus.FAIL,
    "NOT_EXECUTED": JobStatus.SKIP,
    "PAUSED_PENDING_INPUT": JobStatus.SKIP,
    "QUEUED": JobStatus.SKIP,
    "UNSTABLE": JobStatus.FAIL,
}

assert "BUILD_URL" in os.environ, "Are you in the Jenkins environment?"
job_link = os.environ["BUILD_URL"]
response = requests.get("{}wfapi".format(job_link), verify=False).json()
domain = "{uri.scheme}://{uri.netloc}/".format(uri=urlparse(job_link))
stages = response["stages"]

final_dict = {}
failed_nodes = []
nodes_dict = {}


def get_jenkins_json(path):
    return requests.get(urljoin(domain, path), verify=False).json()


for stage in stages:
    link = stage["_links"]["self"]["href"]
    stage_name = stage["name"]
    res = requests.get(urljoin(domain, link), verify=False).json()
    nodes = res["stageFlowNodes"]
    for node in nodes:
        nodes_dict[node["id"]] = node
        nodes_dict[node["id"]]["stageName"] = stage_name


def get_node_full_name(node, node_dict):
    name = ""
    while "parentNodes" in node:
        name = name + "/" + node["name"]
        id = node["parentNodes"][0]
        if id in nodes_dict:
            node = node_dict[id]
        else:
            break
    return name


for key, node in nodes_dict.items():
    logs = get_jenkins_json(node["_links"]["log"]["href"]).get("text", "")
    node_name = node["name"]
    if "Post Actions" in node["stageName"]:
        continue
    node_status = node["status"]
    id = node["id"]
    full_name = get_node_full_name(node, nodes_dict)
    final_dict["{}_{}/{}".format(id, node["stageName"], full_name)] = {
        "status": JENKINS_STATUS_MAPPING[node_status],
        "logs": logs,
    }

JOB_NAME = os.getenv("JOB_NAME")
BUILD_NUMBER = os.getenv("BUILD_NUMBER")
BUILD_ID = os.getenv("BUILD_ID")

prefix = f"https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/logs_dir/"


@pytest.mark.parametrize("test_name", final_dict)
def test_generate_report(test_name):
    os.makedirs("./logs_dir/", exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".log", dir="./logs_dir/"
    )
    tmp.write(final_dict[test_name]["logs"])
    filename = Path(tmp.name).name
    # print(final_dict[test_name]["logs"])
    print("Log path: {}".format(prefix + filename))

    if final_dict[test_name]["status"] == JobStatus.FAIL:
        pytest.fail(
            "Test failed. Please see the log at {}".format(prefix + filename)
        )
    elif final_dict[test_name]["status"] == JobStatus.SKIP:
        pytest.skip(
            "Test skipped. Please see the log at {}".format(prefix + filename)
        )
