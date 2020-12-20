import time
import boto3
import argparse


def pass_env():
    env = "export INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id) \n" + \
        "export INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type) \n" +\
        "export DOCKER_LOG_OPT=(--log-driver awslogs --log-opt awslogs-region=us-west-2 --log-opt awslogs-group=/aws/ec2/dgl-ci --log-opt awslogs-stream=$INSTANCE_ID-docker) "
    import os
    for env_name in ['GIT_COMMIT', 'GIT_URL', 'GIT_BRANCH']:
        if env_name in os.environ:
            env += "export {env_name}={env};\n".format(
                env=os.environ[env_name], env_name=env_name)
    return env

instance_type = 'c5.2xlarge'
disk_size = 150
command = r"""
exec > >(tee /var/log/user-exec.log) 2>&1
{set_env}
docker pull public.ecr.aws/s1o7b3d9/dgl-ci-gpu:conda
cd ~
echo $PWD
sleep 5
""".format(set_env=pass_env())

log_config = r"""
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/cloud-init-output.log",
                        "log_group_name": "/aws/ec2/dgl-ci",
                        "log_stream_name": "{instance_id}-cloudinit"
                    },
                    {
                        "file_path": "/var/log/user-exec.log",
                        "log_group_name": "/aws/ec2/dgl-ci",
                        "log_stream_name": "{instance_id}-exec"
                    }
                ]
            }
        }
    }
}
"""


user_data = r"""
#cloud-config
disable_root: false

runcmd:
 - [ sh, -c, "wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb" ]
 - [ sh, -c, "sudo dpkg -i -E ./amazon-cloudwatch-agent.deb"]
 - "echo '{command}' > /tmp/run.sh"
 - "echo '{log_config}' > /tmp/log.json"
 - [ sh, -c, "sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a append-config -m ec2 -s -c file:/tmp/log.json"]
 - [ bash, "/tmp/run.sh" ]
 - "sleep 10"
 - [ sh, -c, "sudo shutdown -h now"]
""".format(command=command.replace("\n", r"\n").replace("\"", r"\""), log_config=log_config.replace("\n", r"\n").replace("\"", r"\""))
print("==================User Data==================")
print(user_data)
print("==================Data Done==================")

ec2 = boto3.resource('ec2', region_name='us-west-2')
ec2_config = {
    "ImageId": 'ami-01897afb53ff4ec82',  # DL Base AMI 32.0 Ubuntu 18.04
    "InstanceType": instance_type,
    "MaxCount": 1,
    "MinCount": 1,
    "UserData": user_data,
    "KeyName": 'DGL_keypair',
    "TagSpecifications": [{
        "ResourceType": "instance",
        "Tags": [{
            "Key": "Name", "Value": "dgl-ci-ec2-launch"
        }, {
            "Key": "dgl-ci", "Value": "true"
        }, ]
    }],
    "BlockDeviceMappings": [
        {
            'DeviceName': '/dev/sda1',
            'Ebs': {
                'DeleteOnTermination': True,
                'VolumeSize': disk_size,
                'VolumeType': 'gp2'
            },
        },
    ],
    'SecurityGroups': [
        'default', 'SSH group'
    ],
    "InstanceInitiatedShutdownBehavior": 'terminate',
    "IamInstanceProfile": {
        "Name": "dgl_ci_worker_role"
    }
}

instances = ec2.create_instances(**ec2_config)
instance = instances[0]
print("Succeed to launch {}".format(instance))
while (instance.public_ip_address is None):
    instance.reload()
    time.sleep(3)
print(instance.public_ip_address)

while (instance.state['Name'] == "running"):
    instance.reload()
    time.sleep(5)
print(instance.state)
