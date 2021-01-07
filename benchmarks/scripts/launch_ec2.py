import re
import time
import boto3
import argparse
import os
import textwrap
import botocore


def pass_env(extra_env_var):
    env = "export INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id) \n" + \
        "export INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type) \n" +\
        "export DOCKER_LOG_OPT=(--log-driver awslogs --log-opt awslogs-region=us-west-2 --log-opt awslogs-group=/aws/ec2/dgl-ci --log-opt awslogs-stream=$INSTANCE_ID-docker) "
    for env_name in ['GIT_COMMIT', 'GIT_URL', 'GIT_BRANCH'] + extra_env_var:
        if env_name in os.environ:
            env += "export {env_name}={env};\n".format(
                env=os.environ[env_name], env_name=env_name)
    return env


def git_init(ignore_git):
    if ignore_git:
        command = ""
    else:
        assert "GIT_URL" in os.environ and "GIT_BRANCH" in os.environ,\
            "Need to set GIT_URL, GIT_COMMIT and GIT_BRANCH as environment variable"
        command = """
        cd ~
        git clone {GIT_URL} git_repo
        cd git_repo
        git checkout {GIT_BRANCH}
        git submodule update --init --recursive
        """.format(**os.environ)
    return command


def generate_user_data(command, ignore_git, extra_env_var=[]):
    full_command = r"""
                    exec > >(tee /var/log/user-exec.log) 2>&1
                    {set_env}
                    {git_init}
                    {command}
                    sleep 5
                    """.format(set_env=pass_env(extra_env_var), git_init=git_init(ignore_git), command=command)
    full_command = textwrap.dedent(full_command)
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
    mount_efs = r"""
    - apt-get -y install binutils
    - git clone https://github.com/aws/efs-utils
    - cd efs-utils
    - apt-get -y install nfs-common
    - ./build-deb.sh
    - sudo apt-get -y install ./build/amazon-efs-utils*deb
    - file_system_id_1=fs-dbf2f8de
    - efs_mount_point_1=/mnt/efs/fs1
    - mkdir -p "${efs_mount_point_1}"
    - test -f "/sbin/mount.efs" && printf "\n${file_system_id_1}:/ ${efs_mount_point_1} efs tls,_netdev\n" >> /etc/fstab || printf "\n${file_system_id_1}.efs.us-west-2.amazonaws.com:/ ${efs_mount_point_1} nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport,_netdev 0 0\n" >> /etc/fstab
    - test -f "/sbin/mount.efs" && printf "\n[client-info]\nsource=liw\n" >> /etc/amazon/efs/efs-utils.conf
    - retryCnt=15; waitTime=30; while true; do mount -a -t efs,nfs4 defaults; if [ $? = 0 ] || [ $retryCnt -lt 1 ]; then echo File system mounted successfully; break; fi; echo File system not available, retrying to mount.; ((retryCnt--)); sleep $waitTime; done;
    """
    user_data = r"""
    #cloud-config
    disable_root: false
    package_update: true
    

    runcmd:
    {mount_efs}
    - [ sh, -c, "wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb" ]
    - [ sh, -c, "sudo dpkg -i -E ./amazon-cloudwatch-agent.deb"]
    - "echo '{command}' > /tmp/run.sh"
    - "echo '{log_config}' > /tmp/log.json"
    - [ sh, -c, "sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a append-config -m ec2 -s -c file:/tmp/log.json"]
    - [ bash, "/tmp/run.sh" ]
    - "sleep 10"
    # - [ sh, -c, "sudo shutdown -h now"]
    """.format(command=full_command.replace("\n", r"\n").replace("\"", r"\""),
               log_config=log_config.replace("\n", r"\n").replace("\"", r"\""),
               mount_efs=mount_efs)
    print("==================Script Content==================")
    print(full_command)
    print("==================Content End=====================")
    print("==================User Data==================")
    print(user_data)
    print("==================Data Done==================")
    return user_data


def launch_ec2(userdata, instance_type, disk_size=150):
    ec2 = boto3.resource('ec2', region_name='us-west-2')
    ec2_config = {
        "ImageId": 'ami-01897afb53ff4ec82',  # DL Base AMI 32.0 Ubuntu 18.04
        "InstanceType": instance_type,
        "MaxCount": 1,
        "MinCount": 1,
        "UserData": userdata,
        "KeyName": 'DGL_CI_Worker',
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
        # Instance is terminated when shutoff
        "InstanceInitiatedShutdownBehavior": 'terminate',
        "IamInstanceProfile": {
            "Name": "dgl_ci_worker_role"
        }
    }
    instances = ec2.create_instances(**ec2_config)
    instance = instances[0]
    print("Succeed to launch {}".format(instance))
    time.sleep(3)
    while (instance.public_ip_address is None):
        try:
            instance.reload()
        except botocore.exceptions.ClientError:
            print("Not ready yet")
        time.sleep(3)
    print(instance.public_ip_address)
    return instance


def main(args):
    if (args.script_file is not None):
        with open(args.script_file, "r") as f:
            command = f.read()
    else:
        raise Exception("Please specify script file or command")
    instance_type = args.instance_type
    disk_size = args.disk_size
    env_list = [env for env in args.env_vars.split(",") if len(env) > 0]
    instance = launch_ec2(generate_user_data(
        command, args.ignore_git, env_list), instance_type, disk_size)
    while (instance.state['Name'] not in ["stopped", "terminated"]):
        instance.reload()
        time.sleep(5)
    print("Finished Running")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='launch EC2', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--instance-type", default="c5.2xlarge",
                        help="specify instance type")
    parser.add_argument("--ignore-git", action="store_true",
                        default=False, help="whether to skip initialize git repo")
    parser.add_argument("--disk-size", default=150, type=int,
                        help="specify disk size")
    parser.add_argument(
        "--script-file", help="Bash Script to be run", required=True)
    parser.add_argument("--env-vars", default="", help="Extra environment variable to be passed. Split by comma."
                        " i.e. JENKINS_URL, OMP_NUM_THREAD")
    args = parser.parse_args()

    main(args)
