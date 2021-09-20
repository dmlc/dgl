#!/bin/bash

function print_vars {
  for VAR in ${!CCL*} ${!I_MPI*} ${!i_mpi*} ${!KMP_*} ${!OMP_*} LD_PRELOAD ${!DLRM_*} ${!PYTORCH_*} ${!PCL_*} VIRTUAL_ENV ${!ARGS_*} $@ ; do
    if ! test -z ${!VAR} ; then 
       echo "Using $VAR=${!VAR}"
    fi
  done
}

while (( "$#" )); do
  case "$1" in
    -n|-np)
      ARGS_NTASKS=$2
      shift 2
      ;;
    -ppn)
      ARGS_PPN=$2
      shift 2
      ;;
    -f)
      ARGS_HOSTFILE=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      break
      ;;
  esac
done

NNODES=1
NP=1

if ! test -z $SLURM_JOB_ID ; then
  PREFIX="srun -n 1 -N 1 "
else
  PREFIX=
fi

if ! test -z $SLURM_NNODES ; then NNODES=$SLURM_NNODES ; fi
if ! test -z $SLURM_NTASKS ; then NP=$SLURM_NTASKS ; fi
if ! test -z $ARGS_NTASKS ; then NP=$ARGS_NTASKS ; fi
if ! test -z $ARGS_HOSTFILE ; then 
  if ! test -f $ARGS_HOSTFILE ; then
    echo "Hostfile $ARGS_HOSTFILE does not exist!" ; exit 1
  else
    NNODES=`cat $ARGS_HOSTFILE | sort -u | wc -l`
    OPT_HOSTFILE="-f $ARGS_HOSTFILE"
    PREFIX="mpiexec.hydra -np 1 -f $ARGS_HOSTFILE"
  fi
fi

if ! test -z $ARGS_PPN ; then
  OPT_PPN="-ppn $ARGS_PPN" 
  REAL_NNODES=$(( (NP + ARGS_PPN - 1) / ARGS_PPN ))
  if [[ $REAL_NNODES -lt $NNODES ]] ; then NNODES=$REAL_NNODES ; fi
fi

echo "Running $NP tasks on $NNODES nodes"

NUM_THREADS=`$PREFIX lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
NUM_SOCKETS=`$PREFIX lscpu | grep "Socket(s):" | awk '{print $NF}'`
NUM_NUMA_NODES=`$PREFIX lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
THREADS_PER_CORE=`$PREFIX lscpu | grep "Thread(s) per core:" | awk '{print $NF}'`
PHYSCPU=0-$(( NUM_THREADS - 1 ))

if ! test -z $ARGS_HOSTFILE && test -z $ARGS_PPN; then
  OPT_PPN="-ppn $NUM_SOCKETS"
fi

if [ $NP == 1 ] || [ "x$DLRM_USE_MPI" != "x" ] ; then 
    export CCL_WORKER_COUNT=0
else
    if [ "x${CCL_WORKER_COUNT}" == "x" ] ; then
        export CCL_WORKER_COUNT=2
        #export CCL_WORKER_COUNT=2
        #echo "CCL count: ", ${CCL_WORKER_COUNT}
    fi
fi
echo "CCL count: ",${CCL_WORKER_COUNT}
CCL_WORKER_AFFINITY=""

PROC_MASK=$(( ( ( 1 << ( NUM_THREADS - CCL_WORKER_COUNT ) ) - 1 ) << CCL_WORKER_COUNT ))
ZERO_MASK_STR=`printf "%0*X" $(( NUM_THREADS / 4 )) 0`
PROC_MASK_STR=`printf "%X" $PROC_MASK`
MASKS=( )
for(( I=0; I < NUM_SOCKETS; I++)) ; do
  SMASK=""
  for(( J=0; J < NUM_SOCKETS; J++)) ; do
    if [ $J == $I ] ; then SMASK="${PROC_MASK_STR}${SMASK}" ; else SMASK="${ZERO_MASK_STR}${SMASK}" ; fi
  done
  MASKS[$I]="0x$SMASK"
  for((P=0;P < CCL_WORKER_COUNT ; P++)); do CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $(( I * NUM_THREADS + P ))" ; done
done

echo "Affinity: ", ${CCL_WORKER_AFFINITY}
#echo $ZERO_MASK_STR $PROC_MASK_STR
export I_MPI_PIN_DOMAIN=[`echo ${MASKS[@]} | tr " " ","`]
export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`
export OMP_NUM_THREADS=$(( NUM_THREADS - CCL_WORKER_COUNT ))

which python icc gcc mpicc mpiexec.hydra 2> /dev/null

echo "#### INITIAL ENV ####"
print_vars
echo "#### INITIAL ENV ####"

echo "PyTorch version: `python -c "import torch; print(torch.__version__)" 2> /dev/null`"

if ! test -z $SLURM_JOB_ID ; then
srun hostname | sort -u
fi

export MASTER_PORT=12345
export MASTER_ADDR=`$PREFIX hostname`
echo "MASTER_ADDR=$MASTER_ADDR"

CMD=$1
shift
ARGS="$@"

MPIEXE_ARGS="-np $NP $OPT_PPN $OPT_HOSTFILE -l -genv I_MPI_PIN_DOMAIN=$I_MPI_PIN_DOMAIN -genv CCL_WORKER_AFFINITY=$CCL_WORKER_AFFINITY -genv CCL_WORKER_COUNT=$CCL_WORKER_COUNT -genv OMP_NUM_THREADS=$OMP_NUM_THREADS "

echo "Running mpiexec.hydra ${MPIEXE_ARGS} $CMD $@"
eval set -- "${MPIEXE_ARGS} hostname"
mpiexec.hydra $@ | sort 
eval set -- "${MPIEXE_ARGS} $CMD $ARGS"
echo "Running mpiexec.hydra $@"
echo "Start Time:  `date`"
#mpiexec.hydra ${MPIEXE_ARGS} ${CMD} $@
mpiexec.hydra $@
echo "End Time:    `date`"

