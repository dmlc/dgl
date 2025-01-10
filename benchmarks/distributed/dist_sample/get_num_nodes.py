import os
import json

config = json.loads(os.environ["TF_CONFIG"])
print(len(config['cluster']['ps']))
