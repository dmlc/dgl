"""Launching tool for DGL distributed training"""
import os
import stat
import sys
import subprocess
import argparse
import signal
import logging
import time
from threading import Thread

def execute_remote(cmd, ip):
    """execute command line on remote machine via ssh"""
    cmd = 'ssh -o StrictHostKeyChecking=no ' + ip + ' \'' + cmd + '\''
    # thread func to run the job
    def run(cmd):
        subprocess.check_call(cmd, shell = True)

    thread = Thread(target = run, args=(cmd,))
    thread.setDaemon(True)
    thread.start()

def submit_jobs(args):
    """Submit distributed jobs (server and client processes) via ssh"""
    hosts = []
    server_count_per_machine = 0
    ip_config = args.workspace + '/' + args.ip_config
    with open(ip_config) as f:
        for line in f:
            ip, port, count = line.strip().split(' ')
            port = int(port)
            count = int(count)
            server_count_per_machine = count
            hosts.append((ip, port))
    assert args.num_client % len(hosts) == 0
    client_count_per_machine = int(args.num_client / len(hosts))
    # launch server tasks
    server_cmd = 'DGL_ROLE=server'
    server_cmd = server_cmd + ' ' + 'DGL_NUM_CLIENT=' + str(args.num_client)
    server_cmd = server_cmd + ' ' + 'DGL_CONF_PATH=' + str(args.conf_path)
    server_cmd = server_cmd + ' ' + 'DGL_IP_CONFIG=' + str(args.ip_config)
    for i in range(len(hosts)*server_count_per_machine):
        ip, _ = hosts[int(i / server_count_per_machine)]
        cmd = server_cmd + ' ' + 'DGL_SERVER_ID=' + str(i)
        cmd = cmd + ' ' + args.udf_command
        cmd = 'cd ' + str(args.workspace) + '; ' + cmd
        print(cmd)
        execute_remote(cmd, ip)
    print("----------")
    # launch client tasks
    client_cmd = 'DGL_ROLE=client'
    client_cmd = client_cmd + ' ' + 'DGL_IP_CONFIG=' + str(args.ip_config)

    client_cmd = client_cmd + ' ' + 'python3 -m torch.distributed.launch'
    client_cmd = client_cmd + ' ' + '--nproc_per_node=' + str(client_count_per_machine)
    client_cmd = client_cmd + ' ' + '--nnodes=' + str(len(hosts))
    client_cmd = client_cmd + ' ' + '--node_rank=0'
    client_cmd = client_cmd + ' ' + '--master_addr=' + str(hosts[0][0])
    client_cmd = client_cmd + ' ' + '--master_port=1200'

    for i in range(args.num_client):
        node_id = int(i / client_count_per_machine)
        ip, _ = hosts[node_id]
        cmd = client_cmd.replace('node_rank=0', 'node_rank='+str(node_id))
        cmd = cmd + ' ' + args.udf_command
        cmd = cmd + ' ' + '-m torch.distributed.launch'
        cmd = cmd + ' ' + '--nproc_per_node=' + str(client_count_per_machine)
        cmd = cmd + ' ' + '--nnodes=' + str(len(hosts))
        cmd = cmd + ' ' + '--node_rank=' + str(node_id)
        cmd = cmd + ' ' + '--master_addr=' + str(hosts[0][0])
        cmd = cmd + ' ' + '--master_port=1200'
        cmd = 'cd ' + str(args.workspace) + '; ' + cmd
        print(cmd)
        execute_remote(cmd, ip)

    while True:
        time.sleep(10)

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--workspace', type=str,
                        help='Path of user directory of distributed tasks. \
                        This is used to specify a destination location where \
                        the contents of current directory will be rsyncd')
    parser.add_argument('--num_client', type=int, 
                        help='Total number of client processes in the cluster')
    parser.add_argument('--conf_path', type=str, 
                        help='The path to the partition config file. This path can be \
                        a remote path like s3 and dgl will download this file automatically')
    parser.add_argument('--ip_config', type=str, 
                        help='The file for IP configuration for server processes')
    parser.add_argument('--udf_command', type=str,
                        help='User-defined command line')
    args, udf_command = parser.parse_known_args()
    assert len(udf_command) == 1, 'Please provide user command line.'
    args.udf_command = udf_command[0]
    submit_jobs(args)

def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()