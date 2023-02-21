import os

import requests

JOB_NAME = os.getenv("JOB_NAME")
BUILD_NUMBER = os.getenv("BUILD_NUMBER")
BUILD_ID = os.getenv("BUILD_ID")
COMMIT = os.getenv("GIT_COMMIT")

job_link = os.environ["BUILD_URL"]
response = requests.get("{}wfapi".format(job_link), verify=False).json()

# List of status of entire job.
# https://javadoc.jenkins.io/hudson/model/Result.html
status = response["status"]
if status == "SUCCESS":
    status_output = "✅ CI test succeeded"
elif status == "ABORTED":
    status_output = "⚪️ CI test aborted"
else:
    for v in response["stages"]:
        # List of status of individual stage.
        # https://javadoc.jenkins.io/plugin/pipeline-graph-analysis/org/jenkinsci/plugins/workflow/pipelinegraphanalysis/GenericStatus.html
        if v["status"] in ["FAILED", "ABORTED"]:
            stage = v["name"]
            status_output = f"❌ CI test [{status}] in Stage [{name}]."
            break

comment = f"""
Commit ID: {COMMIT}\n
Build ID: {BUILD_ID}\n
Status: {status_output} \n
Report path: [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/report.html) \n
Full logs path: [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/cireport.log)
"""

print(comment)
