import argparse
import os

import requests

parser = argparse.ArgumentParser()
parser.add_argument(
    "--result",
    type=str,
    default="FAILURE",
)
args = parser.parse_args()

JOB_NAME = os.getenv("JOB_NAME")
BUILD_NUMBER = os.getenv("BUILD_NUMBER")
BUILD_ID = os.getenv("BUILD_ID")
COMMIT = os.getenv("GIT_COMMIT")

# List of status of entire job.
# https://javadoc.jenkins.io/hudson/model/Result.html
if args.result == "SUCCESS":
    status_output = "✅ CI test succeeded."
elif args.result == "NOT_BUILT":
    status_output = "⚪️ CI test cancelled due to overrun."
elif args.result in ["FAILURE", "ABORTED"]:
    status_output = "❌ CI test failed."
    JOB_LINK = os.environ["BUILD_URL"]
    response = requests.get("{}wfapi".format(JOB_LINK), verify=False).json()
    for stage in response["stages"]:
        # List of status of individual stage.
        # https://javadoc.jenkins.io/plugin/pipeline-graph-analysis/org/jenkinsci/plugins/workflow/pipelinegraphanalysis/GenericStatus.html
        if stage["status"] in ["FAILED", "ABORTED"]:
            stage_name = stage["name"]
            status_output = f"❌ CI test failed in Stage [{stage_name}]."
            break
else:
    status_output = f"[Debug Only] CI test with result [{args.result}]."


comment = f"""
Commit ID: {COMMIT}\n
Build ID: {BUILD_ID}\n
Status: {status_output}\n
Report path: [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/report.html)\n
Full logs path: [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/cireport.log)
"""

print(comment)
