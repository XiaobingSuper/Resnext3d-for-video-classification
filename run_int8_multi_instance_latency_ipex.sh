#!/bin/sh

###############################################################################
### How to run?
### 1) int8 inference
###    bash run_int8_multi_instance_ipex.sh DATA_PATH dnnl int8 jit resnext3d_configure_jit.json
### 2) fp32 infenence
###    bash run_int8_multi_instance_ipex.sh DATA_PATH dnnl fp32 jit
###
###############################################################################

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CONFIG_FILE=""
ARGS=""

ARGS="$ARGS $1"
echo "### dataset path: $2"

if [ "$2" == "dnnl" ]; then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

if [ "$3" == "int8" ]; then
    ARGS="$ARGS --int8"
    CONFIG_FILE="$CONFIG_FILE --configure-dir $5"
    echo "### running int8 datatype"
else
    echo "### running fp32 datatype"
fi

if [ "$4" == "jit" ]; then
    ARGS="$ARGS --jit"
    echo "### running jit fusion path"
else
    echo "### running jit fusion path"
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=4

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

BATCH_SIZE=1

export OMP_NUM_THREADS=$CORES_PER_INSTANCE
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
for i in $(seq 1 $LAST_INSTANCE); do
    numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
    start_core_i=`expr $i \* $CORES_PER_INSTANCE`
    end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
    LOG_i=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt

    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u main.py -e $ARGS \
        --ipex --dummy -j 0 -be $BATCH_SIZE $CONFIG_FILE 2>&1 | tee $LOG_i &
done

numa_node_0=0
start_core_0=0
end_core_0=`expr $CORES_PER_INSTANCE - 1`
LOG_0=inference_cpu_bs${BATCH_SIZE}_ins0.txt

echo "### running on instance 0, numa node $numa_node_0, core list {$start_core_0, $end_core_0}...\n\n"
numactl --physcpubind=$start_core_0-$end_core_0 --membind=$numa_node_0 python -u main.py -e $ARGS \
    --ipex --dummy -j 0 -be $BATCH_SIZE $CONFIG_FILE 2>&1 | tee $LOG_0

sleep 20
echo -e "\n\n Sum sentences/s together:"
for i in $(seq 0 $LAST_INSTANCE); do
    log=inference_cpu_bs${BATCH_SIZE}_ins${i}.txt
    tail -n 2 $log
done

