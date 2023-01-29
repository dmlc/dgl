import os

import requests

JOB_NAME = os.getenv("JOB_NAME")
BUILD_NUMBER = os.getenv("BUILD_NUMBER")
BUILD_ID = os.getenv("BUILD_ID")
COMMIT = os.getenv("GIT_COMMIT")

job_link = os.environ["BUILD_URL"]
response = requests.get("{}wfapi".format(job_link), verify=False).json()
status = "✅ CI test succeeded"
for v in response["stages"]:
    if v["status"] in ["FAILED", "ABORTED"]:
        status = "❌ CI test failed in Stage [{}].".format(v["name"])
        break

comment = f""" Commit ID: {COMMIT}\n
Build ID: {BUILD_ID}\n
Status: {status} \n
Report path: [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/report.html) \n
Full logs path: [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/cireport.log)
"""

print(comment)
