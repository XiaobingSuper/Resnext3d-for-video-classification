##############################################################################
#### 1) int8 calibration step(non fusion path using ipex):
####    bash run_inference_ipex.sh DATA_PATH dnnl int8 no_jit resnext3d_configure.json calibration
#### 2) int8 inference step(none fusion path using ipex):
####    bash run_inference_ipex.sh DATA_PATH dnnl int8 no_jit resnext3d_configure.json
#### 3) fp32 inference step(non fusion path using ipex):
####    bash run_inference_ipex.sh DATA_PATH dnnl fp32 no_jit

#### 1) int8 calibration step(fusion path using ipex):
####    bash run_inference_ipex.sh DATA_PATH dnnl int8 jit resnext3d_configure_jit.json calibration
#### 2) int8 inference step(fusion path using ipex):
####    bash run_inference_ipex.sh DATA_PATH dnnl int8 jit resnext3d_configure_jit.json
#### 3) fp32 inference step(fusion path using ipex):
####    bash run_inference_ipex.sh  DATA_PATH dnnl fp32 jit
###############################################################################
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

BATCH_SIZE=10

CONFIG_FILE=""

ARGS=""

ARGS="$ARGS $1"
echo "### dataset path: $1"

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
    echo "### running non jit fusion path"
fi

if [ "$6" == "calibration" ]; then
    BATCH_SIZE=5
    ARGS="$ARGS --calibration"
    echo "### running int8 calibration"
fi


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n\n"
sleep 3


python -u main.py -e $ARGS --ipex -j 0 -be $BATCH_SIZE $CONFIG_FILE
