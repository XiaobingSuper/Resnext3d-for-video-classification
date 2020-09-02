#!/usr/bin/env python
# coding: utf-8

from classy_vision.dataset import build_dataset
from classy_vision.models import build_model
from classy_vision.meters import build_meters, AccuracyMeter, VideoAccuracyMeter
from classy_vision.tasks import ClassificationTask
from classy_vision.optim import build_optimizer
from classy_vision.losses import build_loss
from classy_vision.trainer import LocalTrainer
from classy_vision.hooks import (
    CheckpointHook,
    ProgressBarHook,
    LossLrMeterLoggingHook,
)

import torch
from torch.utils import mkldnn as mkldnn_utils
import model_config
from mkldnn_fully_convolutional_linear_head import MkldnnFullyConvolutionalLinear

import argparse
import shutil
import time
import os

import intel_pytorch_extension as ipex

parser = argparse.ArgumentParser(description='PyTorch Video UCF101 Training')
parser.add_argument('video_dir', metavar='DIR',
                    help='path to video files')
parser.add_argument('--num_epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-bt', '--batch-size-train', default=16, type=int,
                    metavar='N',
                    help="bathch size of for training setp")
parser.add_argument('-be', '--batch-size-eval', default=10, type=int,
                    metavar='N',
                    help="bathch size of for eval setp")
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--num-workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('--skip-tensorboard', action='store_true', default=False,
                    help='disable tensorboard')

parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
parser.add_argument('--dnnl', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension auto dnnl path')
parser.add_argument('--int8', action='store_true', default=False,
                    help='enable ipex int8 path')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable ipex jit fusionpath')
parser.add_argument('--calibration', action='store_true', default=False,
                    help='doing calibration step')
parser.add_argument('--configure-dir', default='configure.json', type=str, metavar='PATH',
                    help = 'path to int8 configures, default file name is configure.json')
parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('-w', '--warmup-iterations', default=5, type=int, metavar='N',
                    help='number of warmup iterati ons to run')
def main():
    args = parser.parse_args()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda and args.ipex and args.dnnl:
        assert False, "We can not runing this work on GPU backend and dnnl backend \
                please set one backend.\n"

    if args.cuda:
        print("Using GPU backend to do this work.\n")
    if args.ipex:
        print("Runing on ipex.\n")
        if args.dnnl:
            ipex.core.enable_auto_dnnl()
        else:
            ipex.core.disable_auto_dnnl()
    else:
        if args.dnnl or args.int8:
            print("Please first enable ipex before running dnnl backend or int8 path\n")
    # set it to the folder where video files are saved
    video_dir = args.video_dir + "/UCF-101"
    # set it to the folder where dataset splitting files are saved
    splits_dir = args.video_dir + "/ucfTrainTestlist"
    # set it to the file path for saving the metadata
    #metadata_file = args.video_dir + "/metadata.pth"
    metadata_file = "metadata.pth"

    resnext3d_configs =model_config.ResNeXt3D_Config(video_dir, splits_dir, metadata_file, args.num_epochs)
    resnext3d_configs.setUp()

    datasets = {}
    dataset_train_configs = resnext3d_configs.dataset_configs["train"]
    dataset_test_configs = resnext3d_configs.dataset_configs["test"]
    dataset_train_configs["batchsize_per_replica"] = args.batch_size_train
    # For testing, batchsize per replica should be equal to clips_per_video
    dataset_test_configs["batchsize_per_replica"] = args.batch_size_eval
    dataset_test_configs["clips_per_video"] = args.batch_size_eval

    datasets["train"] = build_dataset(dataset_train_configs)
    datasets["test"] = build_dataset(dataset_test_configs)

    model = build_model(resnext3d_configs.model_configs)
    meters = build_meters(resnext3d_configs.meters_configs)
    loss = build_loss({"name": "CrossEntropyLoss"})
    optimizer = build_optimizer(resnext3d_configs.optimizer_configs)

    # print(model)
    if args.evaluate:
        if args.ipex:
            print("using ipex model to do inference\n")
            model = model.to(device = 'dpcpp:0')
            if args.int8:
                 if args.calibration:
                      ipex.enable_auto_mix_precision(torch.uint8)
                 else:
                     ipex.enable_auto_mix_precision(torch.uint8, configure_file=args.configure_dir)
        validata(datasets, model, loss, meters, args)
        return

    train(datasets, model, loss, optimizer, meters, args)

def train(datasets, model, loss, optimizer, meters, args):
    task = (ClassificationTask()
            .set_num_epochs(args.num_epochs)
            .set_loss(loss)
            .set_model(model)
            .set_optimizer(optimizer)
            .set_meters(meters))
    for phase in ["train", "test"]:
        task.set_dataset(datasets[phase], phase)

    hooks = [LossLrMeterLoggingHook(log_freq=args.print_freq)]
    # show progress
    hooks.append(ProgressBarHook())
    if not args.skip_tensorboard:
        try:
            from tensorboardX import SummaryWriter
            tb_writer = SummaryWriter(log_dir=args.video_dir + "/tensorboard")
            hooks.append(TensorboardPlotHook(tb_writer))
        except ImportError:
            print("tensorboardX not installed, skipping tensorboard hooks")

    checkpoint_dir = f"{args.video_dir}/checkpoint/classy_checkpoint_{time.time()}"
    os.mkdir(checkpoint_dir)
    hooks.append(CheckpointHook(checkpoint_dir, input_args={}))

    task = task.set_hooks(hooks)
    trainer = LocalTrainer(use_gpu=args.cuda, num_dataloader_workers=args.num_workers)
    trainer.train(task)

def validata(datasets, model, loss, meters, args):
    '''
    # This can run eval, but can not get runing time for given iteration
    # so we maually runing the forward step
    task.prepare(use_gpu=args.cuda)
    task.advance_phase() # will get train step
    task.advance_phase() # will get test step
    local_variables = {}

    task.eval_step(use_gpu = args.cuda, local_variables = local_variables)
    '''
    print("Running evaluation step.\n")
    iterator = datasets["test"].iterator(shuffle_seed=0,
                                         epoch=0,
                                         num_workers=args.num_workers,
                                         pin_memory=False,
                                         multiprocessing_context=None)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    #top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(iterator),
                             [batch_time, data_time, losses],
                             prefix='Test: ')

    model = model.eval()
    if args.cuda:
        model = model.cuda()
    if args.ipex and args.int8 and args.calibration:
        with torch.no_grad():
            with ipex.int8_calibration(args.configure_dir):
                end = _time(args.cuda)
                for i, sample in enumerate(iterator):
                    data_time.update(_time(args.cuda) - end)
                    inputs = sample["input"]
                    target = sample["target"]
                    inputs["video"] = inputs["video"].to(device = 'dpcpp:0')
                    inputs["audio"] = inputs["audio"].to(device = 'dpcpp:0')
                    target = target.to(device = 'dpcpp:0')
                    if args.jit:
                        if i == 0:
                            trace_model = torch.jit.trace(model, inputs)
                        output = trace_model(inputs)
                    else:
                        output = model(inputs)

                    loss_data = loss(output, target)
                    batch_time.update(_time(args.cuda) - end)
                    end = _time(args.cuda)

                    if i % args.print_freq == 0:
                        progress.display(i)

                    if i == 10:
                        break
                    ipex.calibration_reset()
    else:
        if args.ipex:
            if args.int8:
                print("running int8 evalation step\n")
            else:
                print("running fp32 evalation step\n")
        if args.dummy:
            print("Using dummpy input to test the performance\n")
            inputs = {}
            inputs["video"] = torch.randn(args.batch_size_eval, 3, 32, 128, 170)
            inputs["audio"] = torch.randn(args.batch_size_eval, 0, 1)
            target = torch.arange(1, args.batch_size_eval + 1).long()
            if args.ipex:
                inputs["video"] = inputs["video"].to(device = 'dpcpp:0')
                inputs["audio"] = inputs["audio"].to(device = 'dpcpp:0')
                target = target.to(device = 'dpcpp:0')
            if args.cuda:
                inputs["video"] = inputs["video"].cuda()
                inputs["audio"] = inputs["audio"].cuda()
                target = target.cuda()

            number_iter = len(iterator)
            with torch.no_grad():
                for i in range(number_iter):
                    #data_time.update(_time(args.cuda) - end)
                    if i >= args.warmup_iterations:
                        end = time.time()
                    if args.jit:
                        if i == 0:
                            trace_model = torch.jit.trace(model, inputs)
                        output = trace_model(inputs)
                    else:
                        output = model(inputs)

                    #loss_data = loss(output, target)
                    # TODO get accuracy
                    # for meter in meters:
                    #    meter.update(output, target, is_train=False)

                    if i >= args.warmup_iterations:
                        batch_time.update(_time(args.cuda) - end)

                    if i % args.print_freq == 0:
                        progress.display(i)
                    if i == 100:
                        break
                batch_size = args.batch_size_eval
                latency = batch_time.avg / batch_size * 1000
                perf = batch_size / batch_time.avg
                print('inference latency %.2f ms'%latency)
                print('inference performance %.2f fps'%perf)
        else:
            with torch.no_grad():
                end = _time(args.cuda)
                for i, sample in enumerate(iterator):
                    data_time.update(_time(args.cuda) - end)
                    inputs = sample["input"]
                    target = sample["target"]
                    if args.cuda:
                        inputs["video"] = inputs["video"].cuda()
                        inputs["audio"] = inputs["audio"].cuda()
                        target = target.cuda()
                    elif args.ipex:
                        inputs["video"] = inputs["video"].to(device = 'dpcpp:0')
                        inputs["audio"] = inputs["audio"].to(device = 'dpcpp:0')
                    if args.jit:
                        if i == 0:
                            trace_model = torch.jit.trace(model, inputs)
                        output = trace_model(inputs)
                    else:
                        output = model(inputs)

                    loss_data = loss(output, target)
                    # TODO get accuracy
                    # for meter in meters:
                    #    meter.update(output, target, is_train=False)

                    batch_time.update(_time(args.cuda) - end)
                    end = _time(args.cuda)

                    if i % args.print_freq == 0:
                        progress.display(i)
                    if i == 30:
                        break
                # TODO
                # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                #      .format(top1=top1, top5=top5))

def _time(use_cuda):
    if use_cuda:
        torch.cuda.synchronize()
    return time.time()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()

