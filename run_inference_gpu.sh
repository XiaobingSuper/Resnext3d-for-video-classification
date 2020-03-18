BATCH_SIZE=10
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  shift
fi

LOG=inference_gpu_bs${BATCH_SIZE}.txt
python -u main.py -e UCF101 \
    --batch-size-eval $BATCH_SIZE \
    2>&1 | tee $LOG_0