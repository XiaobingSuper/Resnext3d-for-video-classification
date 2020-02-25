# Resnext3d-for-video-classification
Using [ClassyVision](https://github.com/facebookresearch/ClassyVision) to implement Resnext3d model, refer to [How to do video classification](https://classyvision.ai/tutorials/video_classification) part, so you can see the details how to do a video classification tast using ClassyVision

## Requirements(suggest install the following package using source code)

- Install [PyTorch](https://github.com/pytorch/pytorch)
- Install [TorchVison](https://github.com/pytorch/vision)
- Install [ClassyVision](https://github.com/facebookresearch/ClassyVision)
- Download the Video dataset: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php): see [How to do video classification](https://classyvision.ai/tutorials/video_classification)
  
## Usage
```
usage: main.py [--num_epochs N] [-e] [--no-cuda] DIR

PyTorch Video UCF101 Training

positional arguments:
  DIR             path to video files

optional arguments:
  -h, --help      show this help message and exit
  --num_epochs N  number of total epochs to run
  -e, --evaluate  evaluate model on validation set
  --no-cuda       disable CUDA

```
