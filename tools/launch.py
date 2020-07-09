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

# thread func to run the job
def run(cmd):
    subprocess.check_call(cmd, shell = True)

def execute(cmd):
    thread = Thread(target = run, args=(cmd,))
    thread.setDaemon(True)
    thread.start()

def kill_python_proc(args):
    """Kill all the python process accross machines"""
    ip_config = args.workspace + '/' + args.ip_config
    with open(args.ip_config) as f:
        for line in f:
            ip, _, _ = line.strip().split(' ')
            cmd = 'pkill -9 python'
            cmd = 'ssh -o StrictHostKeyChecking=no ' + ip + ' \'' + cmd + '\''
            execute(cmd)

    while True:
        time.sleep(10)

def submit_jobs(server_cmd, client_cmd, args):
    """Submit distributed jobs via ssh"""
    ip_addr = []
    server_count_per_machine = 0
    ip_config = args.workspace + '/' + args.ip_config
    with open(args.ip_config) as f:
        for line in f:
            ip, port, count = line.strip().split(' ')
            port = int(port)
            count = int(count)
            server_count_per_machine = count
            ip_addr.append((ip, port))
    client_count_per_machine = args.num_client / len(ip_addr)

    # launch server tasks
    for i in range(len(ip_addr)*server_count_per_machine):
        ip, _ = ip_addr[int(i / server_count_per_machine)]
        cmd = server_cmd + ' --id ' + str(i)
        cmd = 'cd ' + str(args.workspace) + '; ' + cmd
        cmd = 'ssh -o StrictHostKeyChecking=no ' + ip + ' \'' + cmd + '\''
        execute(cmd)
    # launch client tasks
    for i in range(args.num_client):
        ip, _ = ip_addr[int(i / client_count_per_machine)]
        new_node_rank = 'node_rank=' + str(i)
        cmd = client_cmd.replace('node_rank=0', new_node_rank)
        cmd = 'cd ' + str(args.workspace) + '; ' + cmd
        cmd = 'ssh -o StrictHostKeyChecking=no ' + ip + ' \'' + cmd + '\''
        execute(cmd)

    while True:
        time.sleep(10)

def server_command(args):
    server_cmd = 'python3 ' + str(args.exe_file) + ' --server'
    server_cmd = server_cmd + ' --graph-name ' + str(args.graph_name)
    server_cmd = server_cmd + ' --num-client ' + str(args.num_client)
    server_cmd = server_cmd + ' --conf_path ' + str(args.conf_path)
    server_cmd = server_cmd + ' --ip_config ' + str(args.ip_config)
    return server_cmd

def client_command(args, udf_args):
    client_cmd = 'python3 -m torch.distributed.launch'
    client_cmd = client_cmd + ' --nproc_per_node=' + str(args.nproc_per_node)
    client_cmd = client_cmd + ' --nnodes=' + str(args.nnodes)
    client_cmd = client_cmd + ' --node_rank=0'
    client_cmd = client_cmd + ' --master_addr=' + str(args.master_addr)
    client_cmd = client_cmd + ' --master_port=' + str(args.master_port)
    client_cmd = client_cmd + ' ' + str(args.exe_file)
    client_cmd = client_cmd + ' --graph-name ' + str(args.graph_name)
    client_cmd = client_cmd + ' --ip_config ' + str(args.ip_config)
    client_cmd = client_cmd + ' ' + udf_args
    return client_cmd 

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--kill',  action='store_true',
                        help='wether to kill all the python processes')
    parser.add_argument('--workspace', type=str,
                        help='Path of user workspace of distributed tasks')
    parser.add_argument('--exe_file', type=str,
                        help="Python executable")
    parser.add_argument('--server', action='store_true',
                        help='wether this is a server')
    parser.add_argument('--graph-name', type=str, 
                        help='graph name')
    parser.add_argument('--num-client', type=int, 
                        help='The number of clients')
    parser.add_argument('--conf_path', type=str, 
                        help='The path to the partition config file')
    parser.add_argument('--ip_config', type=str, 
                        help='The file for IP configuration')
    parser.add_argument('--nproc_per_node', type=int, default=1,
                        help='Number of processes per node')
    parser.add_argument('--nnodes', type=int, 
                        help='Total number of nodes')
    parser.add_argument('--master_addr', type=str, 
                        help='IP address of master node')
    parser.add_argument('--master_port', type=int, default=1234,
                        help='Port of master node')
    parser.add_argument('--udf_args', type=str,
                        help='user-defined arguments.')
    args, udf_args = parser.parse_known_args()

    if args.kill:
        kill_python_proc(args)
    else:
        server_cmd = server_command(args)
        client_cmd = client_command(args, udf_args[0])
        submit_jobs(server_cmd, client_cmd, args)

def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()