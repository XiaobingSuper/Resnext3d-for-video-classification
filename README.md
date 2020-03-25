# Resnext3d-for-video-classification
Using [ClassyVision](https://github.com/facebookresearch/ClassyVision) to implement Resnext3d model, refer to [How to do video classification](https://classyvision.ai/tutorials/video_classification) part, so you can see the details how to do a video classification task using ClassyVision

## Requirements(suggest install the following package using source code)

- Install [PyTorch](https://github.com/pytorch/pytorch) as steps in [README.md](https://github.com/pytorch/pytorch/blob/master/README.md#installation):

- Install [TorchVison](https://github.com/pytorch/vision):
```
git clone https://github.com/pytorch/vision.git
python setup.py install
```
- Install [ClassyVision](https://github.com/facebookresearch/ClassyVision):
```
git clone https://github.com/facebookresearch/ClassyVision.git
cd ClassyVision
pip install .
```
- Install [PyAV](https://github.com/mikeboers/PyAV): `conda install av -c conda-forge`
- Download the Video dataset: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php): see [How to do video classification](https://classyvision.ai/tutorials/video_classification)
  
## Usage
```
usage: main.py [-h] [--num_epochs N] [-bt N] [-be N] [-p N] [-e] [--no-cuda]
               [--mkldnn]
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
  -e, --evaluate        evaluate model on validation set
  --no-cuda             disable CUDA
  --mkldnn              use mkldnn backend
```
## Note
For 3d ops of MKLDNN backend, PyTorch master not support it now, only available our internal branch. Thanks!

## Performance data logs(test on skx-8180, 2 sockets, 56 threads)
1. Running on native cpu path:
```
### using OMP_NUM_THREADS=56
### using KMP_AFFINITY=granularity=fine,compact,1,0
Namespace(batch_size_eval=10, batch_size_train=16, evaluate=True, mkldnn=False, no_cuda=True, num_epochs=300, print_fr          eq=10, video_dir='UCF101')
Using native CPU backend to do this work.

Running evaluation step.

Test: [   0/3781]       Time 109.408 (109.408)  Data  0.677 ( 0.677)    Loss 0.0000e+00 (0.0000e+00)
Test: [  10/3781]       Time 95.489 (96.458)    Data  0.674 ( 0.622)    Loss 0.0000e+00 (0.0000e+00)
Test: [  20/3781]       Time 94.773 (95.659)    Data  0.547 ( 0.608)    Loss 0.0000e+00 (0.0000e+00)
Test: [  30/3781]       Time 95.197 (95.485)    Data  0.634 ( 0.614)    Loss 0.0000e+00 (0.0000e+00)
```
2. Running on MKLDNN backend path:
```
### cache input/output in mkldnn format
### using OMP_NUM_THREADS=56
### using KMP_AFFINITY=granularity=fine,compact,1,0
Namespace(batch_size_eval=10, batch_size_train=16, evaluate=True, mkldnn=True, no_cuda=True, num_epochs=300, print_freq=10, video_dir='UCF101')
Using MKLDNN backend to do this work.

Running evaluation step.

Test: [   0/3781]       Time  7.979 ( 7.979)    Data  0.689 ( 0.689)    Loss 0.0000e+00 (0.0000e+00)
Test: [  10/3781]       Time  7.831 ( 7.896)    Data  0.697 ( 0.650)    Loss 0.0000e+00 (0.0000e+00)
Test: [  20/3781]       Time  7.103 ( 7.636)    Data  0.616 ( 0.651)    Loss 0.0000e+00 (0.0000e+00)
Test: [  30/3781]       Time  7.152 ( 7.483)    Data  0.643 ( 0.661)    Loss 0.0000e+00 (0.0000e+00)
```
We can get `~13x` performance improvement
