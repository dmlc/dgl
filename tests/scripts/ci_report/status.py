import os
import requests
JOB_NAME = os.getenv("JOB_NAME")
BUILD_NUMBER = os.getenv("BUILD_NUMBER")
BUILD_ID = os.getenv("BUILD_ID")


job_link = os.environ["BUILD_URL"]
response = requests.get('{}wfapi'.format(job_link)).json()
# IN_PROGRESS is expected, sincce the pipeline is not finished when executing this script
success = response["status"] in ['SUCCESS', 'IN_PROGRESS']

if success:
    status = "✅ CI test succeeded"
else:
    status = "❌ CI test failed"


comment = f""" {JOB_NAME}
{status} \n
Report at [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/report.html) \n
Full logs at [link](https://dgl-ci-result.s3.us-west-2.amazonaws.com/{JOB_NAME}/{BUILD_NUMBER}/{BUILD_ID}/logs/cireport.log)
"""

print(comment)