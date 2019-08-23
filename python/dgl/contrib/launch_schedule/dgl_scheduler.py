# DGL scheduler for distributed training. Given the description of server-workers, 
# client-workers, or sampler-workers through command line options, the scheduler 
# will do the following jobs:
#
# 1. Copy the executable files, data files and some scripts to a temporary directory
#    in each remote machine, the temporary directory is specified by user.
#
# 2. Start communictors in each machine by running python script 'worker.py',
#    which will establish communication among machines
#
# 3. Start server-worker or client-worker for each machine.

# To be done