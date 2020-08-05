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

def execute_remote(cmd, ip, thread_list):
    """execute command line on remote machine via ssh"""
    cmd = 'ssh -o StrictHostKeyChecking=no ' + ip + ' \'' + cmd + '\''
    # thread func to run the job
    def run(cmd):
        subprocess.check_call(cmd, shell = True)

    thread = Thread(target = run, args=(cmd,))
    thread.setDaemon(True)
    thread.start()
    thread_list.append(thread)

def submit_jobs(args, udf_command):
    """Submit distributed jobs (server and client processes) via ssh"""
    hosts = []
    thread_list = []
    server_count_per_machine = 0
    ip_config = args.workspace + '/' + args.ip_config
    with open(ip_config) as f:
        for line in f:
            ip, port, count = line.strip().split(' ')
            port = int(port)
            count = int(count)
            server_count_per_machine = count
            hosts.append((ip, port))
    tot_num_clients = args.num_trainers * (1 + args.num_samplers) * len(hosts)
    # launch server tasks
    server_cmd = 'DGL_ROLE=server'
    server_cmd = server_cmd + ' ' + 'DGL_NUM_CLIENT=' + str(tot_num_clients)
    server_cmd = server_cmd + ' ' + 'DGL_CONF_PATH=' + str(args.part_config)
    server_cmd = server_cmd + ' ' + 'DGL_IP_CONFIG=' + str(args.ip_config)
    for i in range(len(hosts)*server_count_per_machine):
        ip, _ = hosts[int(i / server_count_per_machine)]
        cmd = server_cmd + ' ' + 'DGL_SERVER_ID=' + str(i)
        cmd = cmd + ' ' + udf_command
        cmd = 'cd ' + str(args.workspace) + '; ' + cmd
        execute_remote(cmd, ip, thread_list)
    # launch client tasks
    client_cmd = 'DGL_DIST_MODE="distributed" DGL_ROLE=client'
    client_cmd = client_cmd + ' ' + 'DGL_NUM_CLIENT=' + str(tot_num_clients)
    client_cmd = client_cmd + ' ' + 'DGL_CONF_PATH=' + str(args.part_config)
    client_cmd = client_cmd + ' ' + 'DGL_IP_CONFIG=' + str(args.ip_config)
    if os.environ.get('OMP_NUM_THREADS') is not None:
        client_cmd = client_cmd + ' ' + 'OMP_NUM_THREADS=' + os.environ.get('OMP_NUM_THREADS')
    if os.environ.get('PYTHONPATH') is not None:
        client_cmd = client_cmd + ' ' + 'PYTHONPATH=' + os.environ.get('PYTHONPATH')

    torch_cmd = '-m torch.distributed.launch'
    torch_cmd = torch_cmd + ' ' + '--nproc_per_node=' + str(args.num_trainers)
    torch_cmd = torch_cmd + ' ' + '--nnodes=' + str(len(hosts))
    torch_cmd = torch_cmd + ' ' + '--node_rank=' + str(0)
    torch_cmd = torch_cmd + ' ' + '--master_addr=' + str(hosts[0][0])
    torch_cmd = torch_cmd + ' ' + '--master_port=' + str(1234)
    for node_id, host in enumerate(hosts):
        ip, _ = host
        new_torch_cmd = torch_cmd.replace('node_rank=0', 'node_rank='+str(node_id))
        if 'python3' in udf_command:
            new_udf_command = udf_command.replace('python3', 'python3 ' + new_torch_cmd)
        elif 'python2' in udf_command:
            new_udf_command = udf_command.replace('python2', 'python2 ' + new_torch_cmd)
        else:
            new_udf_command = udf_command.replace('python', 'python ' + new_torch_cmd)
        cmd = client_cmd + ' ' + new_udf_command
        cmd = 'cd ' + str(args.workspace) + '; ' + cmd
        execute_remote(cmd, ip, thread_list)

    for thread in thread_list:
        thread.join()

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--workspace', type=str,
                        help='Path of user directory of distributed tasks. \
                        This is used to specify a destination location where \
                        the contents of current directory will be rsyncd')
    parser.add_argument('--num_trainers', type=int,
                        help='The number of trainer processes per machine')
    parser.add_argument('--num_samplers', type=int,
                        help='The number of sampler processes per trainer process')
    parser.add_argument('--part_config', type=str,
                        help='The file (in workspace) of the partition config')
    parser.add_argument('--ip_config', type=str,
                        help='The file (in workspace) of IP configuration for server processes')
    args, udf_command = parser.parse_known_args()
    assert len(udf_command) == 1, 'Please provide user command line.'
    assert args.num_trainers > 0, '--num_trainers must be a positive number.'
    assert args.num_samplers >= 0
    udf_command = str(udf_command[0])
    if 'python' not in udf_command:
        raise RuntimeError("DGL launching script can only support Python executable file.")
    submit_jobs(args, udf_command)

def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()
