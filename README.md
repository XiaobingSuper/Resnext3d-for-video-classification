# Resnext3d-for-video-classification
Using [ClassyVision](https://github.com/facebookresearch/ClassyVision) to implement Resnext3d model, refer to [How to do video classification](https://classyvision.ai/tutorials/video_classification) part, so you can see the details how to do a video classification task using ClassyVision

## Requirements(suggest install the following package using source code)

- Install [PyTorch-extension](https://gitlab.devtools.intel.com/intel-pytorch-extension/ipex-cpu-dev) as steps in [README.md](https://gitlab.devtools.intel.com/intel-pytorch-extension/ipex-cpu-dev#installation):

- Install [TorchVison](https://github.com/pytorch/vision):
```
git clone https://github.com/pytorch/vision.git
git checkout v0.5.1
python setup.py install
```
- Install [ClassyVision](https://github.com/facebookresearch/ClassyVision):
```
git clone https://github.com/facebookresearch/ClassyVision.git
cd ClassyVision
pip install .
```
- Install [PyAV](https://github.com/mikeboers/PyAV): `conda install av -c conda-forge`
- Build jemalloc
```
    cd ..
    git clone  https://github.com/jemalloc/jemalloc.git    
    cd jemalloc 
    ./autogen.sh
    ./configure --prefix=your_path(eg: /home/tdoux/tdoux/jemalloc/)
    make
    make install
```
- Download the Video dataset: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php): see [How to do video classification](https://classyvision.ai/tutorials/video_classification)
  
## Usage
```
usage: main.py [-h] [--num_epochs N] [-bt N] [-be N] [-p N] [-j N] [-e] [--no-cuda] [--skip-tensorboard] [--ipex] [--dnnl]
               [--int8] [--jit] [--calibration] [--configure-dir PATH] [--dummy] [-w N]
               DIR

PyTorch Video UCF101 Training

positional arguments:
  DIR                   path to video files

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs N        number of total epochs to run
  -bt N, --batch-size-train N
                        bathch size of for training setp
  -be N, --batch-size-eval N
                        bathch size of for eval setp
  -p N, --print-freq N  print frequency (default: 10)
  -j N, --num-workers N
                        number of data loading workers (default: 0)
  -e, --evaluate        evaluate model on validation set
  --no-cuda             disable CUDA
  --skip-tensorboard    disable tensorboard
  --ipex                use intel pytorch extension
  --dnnl                enable Intel_PyTorch_Extension auto dnnl path
  --int8                enable ipex int8 path
  --jit                 enable ipex jit fusionpath
  --calibration         doing calibration step
  --configure-dir PATH  path to int8 configures, default file name is configure.json
  --dummy               using dummu data to test the performance of inference
  -w N, --warmup-iterations N
                        number of warmup iterati ons to run
```

## Testing IPEX Performance
#### Pre load Jemalloc for better performance.
```
export LD_PRELOAD= "path/lib/libjemalloc.so"
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```
### FP32 path:
- inference throughput( 1 socket /ins):
```
bash run_int8_multi_instance_ipex.sh /lustre/dataset/UCF101 dnnl fp32 jit
```
- inference realtime( 4 cores /ins):
```
bash run_int8_multi_instance_latency_ipex.sh /lustre/dataset/UCF101 dnnl fp32 jit
```
### INT8 path:
- inference throughput( 1 socket /ins):
```
bash run_int8_multi_instance_ipex.sh /lustre/dataset/UCF101 dnnl int8 jit resnext3d_configure_jit.json
```
- inference realtime( 4 cores /ins):
```
bash run_int8_multi_instance_latency_ipex.sh /lustre/dataset/UCF101 dnnl int8 jit resnext3d_configure_jit.json
```
