#!/bin/sh


###############################################
### cpu runs on two sockets:
### ./run_inference_cpu.sh (throughput)
### ./run_inference_cpu.sh --single (realtime)
###  also add --mkldnn to run MKLDNN backend
###
##############################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

BATCH_SIZE=10
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  TOTAL_CORES=4
  LAST_CORE=`expr $TOTAL_CORES - 1`
  PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
  shift
fi

ARGS=""
if [[ "$1" == "--mkldnn" ]]; then
    ARGS="$ARGS --mkldnn"
    echo "### cache input/output in mkldnn format"
    shift
fi

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
sleep 3

LOG=inference_cpu_bs${BATCH_SIZE}.txt
python -u main.py -e UCF101 \
    --batch-size-eval $BATCH_SIZE \
    --no-cuda $ARGS \
    2>&1 | tee $LOG_0
