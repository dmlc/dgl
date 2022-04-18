import pytest
import json
import enum
from pathlib import Path
import tempfile

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

import requests
from urllib.parse import urlparse,urljoin
import os
assert "BUILD_URL" in os.environ, "Are you in the Jenkins environment?"
job_link = os.environ["BUILD_URL"]
response = requests.get('{}wfapi'.format(job_link)).json()
domain = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(job_link))
stages = response["stages"]

final_dict = {}
failed_nodes = []
# failed_stages = list(filter(lambda x: x['status'] != 'SUCCESS', stages))

def get_jenkins_json(path):
    return requests.get(urljoin(domain, path)).json()

for stage in stages:
    link = stage['_links']['self']['href']
    stage_name = stage['name']
    res = requests.get(urljoin(domain,link)).json()
    nodes = res['stageFlowNodes']
    # failed_nodes = list(filter(lambda x: x['status'] != 'SUCCESS', nodes))
    for node in nodes:
        failed_log = get_jenkins_json(node['_links']['log']['href']).get('text', '')
        node_name = node['name']
        node_status = node['status']
        final_dict["{}/{}".format(stage_name, node_name)] = {            
            "status": JENKINS_STATUS_MAPPING[node_status],
            "logs": failed_log
        }


@pytest.mark.parametrize("test_name", final_dict)
def test_generate_report(test_name):
    os.makedirs("./logs_dir/", exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".log", dir="./logs_dir/")
    tmp.write(final_dict[test_name]["logs"])
    filename = Path(tmp.name).name
    # print(final_dict[test_name]["logs"])
    print("Log path: {}".format(filename))

    if final_dict[test_name]["status"] == JobStatus.FAIL:
        pytest.fail("Test failed. Please see the log at {}".format(filename))
    elif final_dict[test_name]["status"] == JobStatus.SKIP:
        pytest.skip("Test skipped. Please see the log at {}".format(filename))