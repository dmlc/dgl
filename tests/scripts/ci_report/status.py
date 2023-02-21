import os

import requests

JOB_NAME = os.getenv("JOB_NAME")
BUILD_NUMBER = os.getenv("BUILD_NUMBER")
BUILD_ID = os.getenv("BUILD_ID")
COMMIT = os.getenv("GIT_COMMIT")
JOB_LINK = os.environ["BUILD_URL"]
response = requests.get("{}wfapi".format(JOB_LINK), verify=False).json()

# List of status of entire job.
# https://javadoc.jenkins.io/hudson/model/Result.html
status_output = "✅ CI test succeeded"
for stage in response["stages"]:
    # List of status of individual stage.
    # https://javadoc.jenkins.io/plugin/pipeline-graph-analysis/org/jenkinsci/plugins/workflow/pipelinegraphanalysis/GenericStatus.html
    if stage["status"] in ["FAILED", "ABORTED"]:
        stage_name = stage["name"]
        status_output = f"❌ CI test [{status}] in Stage [{stage_name}]."
        break

comment = f"""
Commit ID: {COMMIT}\n
Build ID: {BUILD_ID}\n
Status: {status_output}\n
Report path: [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/report.html)\n
Full logs path: [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/cireport.log)
"""

print(comment)
