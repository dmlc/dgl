# utilites for DGL launch scheduler
import inspect
import logging
import os
import sys
import subprocess
import traceback
import urllib

def config_logging(logfile):
    """ Config global logging interface

    All the messages will be output to log file, the messages with level >=
    logging.INFO will be output console
    """
    format_str = ("%(levelname)-8s %(asctime)s %(filename)s:%(lineno)d"
                  " ] %(message)s")
    logging.basicConfig(filename=logfile, filemode='w',
                        format=format_str, level=logging.DEBUG)
    logger = logging.getLogger()
    format_str = ("%(filename)s: %(message)s")
    formatter = logging.Formatter(format_str)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class SocketWrapper(object):
    """A simple warpper for python socket
    """
    def __init__(self, sockobj):
    	self._buf = ''
    	self._sockobj = sockobj

    def close(self):
        self._sockobj.close()

    def send(self, msg):
        """Send out message
        """
        msg = urllib.quote(msg)
        msg = '%s\n' %msg
        self._sockobj.sendall(mesg)

    def recv(self, msg):
        """ Receive message
        """
        while not '\n' in self._buf:
            self._buf += self._sockobj.recv(1024)
        message, remain = self._buf.split('\n', 1)
        self._buf = remain
        return urllib.unquote(message)


class CmdTool(object):
    """Command line tool, used to

    1. run command in local Bash
    2. run ssh command on remote machine
    3. copy files to remote machines via scp
    """
    def __init__(self, ssh_port=22):
        self._ssh_port = ssh_port

    def display(self, msg):
        """Display message string and caller

        Parameters
        ----------
        msg : str
            Message string
        """
        caller = '<%s.%s>' %(self.__class__.__name__, inspect.stack()[1][3])
        msg = '%s: %s' %(caller, msg)
        print(msg)
        sys.stdout.flush()

    def set_ssh_port(self, ssh_port):
        """Set ssh port

        Parameters
        ----------
        ssh_port : int
            port number
        """
        self._ssh_port = ssh_port;

    def wait_cmd(self, process, cmd_str):
        """Wait for the process running command line.

        Parameters
        ----------
        process : Process
            python process created by CmdTool
        cmd_str : string
            command line string
        """
        retcode = process.wait()
        if retcode != 0:
            msg = 'Fail with retcode(%s): %s' %(retcode, cmd_str)
            raise RuntimeError(msg)

    def run_cmd(self, cmd_str):
        """Run local command

        Parameters
        ----------
        cmd_str : string
            string of command line
        """
        cmd_lines = [line for line in cmd_str.splitlines() if len(line) > 0]
        cmd_str = ' \\\n'.join(cmd_lines)
        os.environ['path'] = '/usr/local/bin:/bin:/usr/bin:/sbin/'
        process = subprocess.Popen(cmd_str, shell=True, env=os.environ)
        msg = 'run command PPID=%s PID=%s CMD=%s' %(os.getpid(), process.pid, cmd_str)
        logging.debug(msg)
        return process

    def run_cmd_and_wait(self, cmd_str):
    	"""Run local command line and wait until finish

    	Parameters
    	----------
    	cmd_str : string
    	    string of command line
    	"""
    	process = self.run_cmd(cmd_str)
    	self.wait_cmd(process, cmd_str)

    def run_ssh_cmd(self, machine, remote_cmd):
        """Run command line via SSH in remote machine

        Parameters
        ----------
        machine : string
            machine ip address
        remote_cmd : string
            string of remote command line
        """
        ssh = 'ssh -q -p %s' % self.ssh_port
        ssh_cmd = '%s %s \'%s\'' %(ssh, machine, remote_cmd)
        return self.run_cmd(ssh_cmd)

    def dispatch_file(self, file_list, dir_dict):
        """Copy source files to remote machines

        Parameters
        ----------
        file_list : list
            a list of filename string
        dir_dict : dict
            a dict of address, the key is machine address, 
            and the value is file path.
        """
        files = ' '.join(file_list)
        job_list = []
        for machine , tmp_dir in dir_dict.items():
            cmd_scp = ('scp -q -P %s %s %s:%s/ >/dev/null'
                       % (self.ssh_port, files, machine, tmp_dir))
            job_list.append([self.run_cmd(cmd_scp), cmd_scp])
        for process, cmd_scp in job_list:
            self.wait_cmd(process, cmd_scp)
