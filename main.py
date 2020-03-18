#!/usr/bin/env python
# coding: utf-8

from classy_vision.dataset import build_dataset
from classy_vision.models import build_model
from classy_vision.meters import build_meters, AccuracyMeter, VideoAccuracyMeter
from classy_vision.tasks import ClassificationTask
from classy_vision.optim import build_optimizer
from classy_vision.losses import build_loss
from classy_vision.trainer import LocalTrainer
from classy_vision.hooks import CheckpointHook
from classy_vision.hooks import LossLrMeterLoggingHook

import torch
from torch.utils import mkldnn as mkldnn_utils
import time
import os
import model_config
import argparse
import shutil


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
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('--mkldnn', action='store_true', default=False,
                    help='use mkldnn backend')

def main():
    args = parser.parse_args()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda and argd.mkldnn:
        assert False, "We can not runing this work on GPU backend and MKLDNN backend \
                please set one backend.\n"

    if args.cuda:
        print("Using GPU backend to do this work.\n")
    elif args.mkldnn:
        print("Using MKLDNN backend to do this work.\n")
    else:
        print("Using native CPU backend to do this work.\n")

    # set it to the folder where video files are saved
    video_dir = args.video_dir + "/UCF-101"
    # set it to the folder where dataset splitting files are saved
    splits_dir = args.video_dir + "/ucfTrainTestlist"
    # set it to the file path for saving the metadata
    metadata_file = args.video_dir + "/metadata.pth"

    resnext3d_configs =model_config.ResNeXt3D_Config(video_dir, splits_dir, metadata_file, args.num_epochs)
    resnext3d_configs.setUp()

    datasets = {}
    dataset_train_config = resnext3d_configs.dataset_configs["train"]
    dataset_test_config = resnext3d_configs.dataset_configs["test"]
    dataset_train_config["batchsize_per_replica"] = args.batch_size_train
    # For testing, batchsize per replica should be equal to clips_per_video
    dataset_test_config["batchsize_per_replica"] = args.batch_size_eval
    dataset_test_config["clips_per_video"] = args.batch_size_eval

    datasets["train"] = build_dataset(dataset_train_config)
    datasets["test"] = build_dataset(dataset_test_config)

    model = build_model(resnext3d_configs.model_configs)
    meters = build_meters(resnext3d_configs.meters_configs)
    loss = build_loss({"name": "CrossEntropyLoss"})
    optimizer = build_optimizer(resnext3d_configs.optimizer_configs)

    #print(model)
    if args.evaluate:
        validata(datasets, model, loss, args)
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

    hooks = [LossLrMeterLoggingHook(log_freq=4)]

    checkpoint_dir = f"/home/xiaobinz/Downloads/resnext3d/ucf/checkpoint/classy_checkpoint_{time.time()}"
    os.mkdir(checkpoint_dir)
    hooks.append(CheckpointHook(checkpoint_dir, input_args={}))

    task = task.set_hooks(hooks)
    trainer = LocalTrainer(use_gpu = args.cuda)
    trainer.train(task)

def validata(datasets, model, loss, args):
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
    iterator = datasets["test"].iterator(shuffle_seed = 0,
                                         epoch= 0,
                                         num_workers=0,  # 0 indicates to do dataloading on the master process
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
    elif args.mkldnn:
        model = mkldnn_utils.to_mkldnn(model)
        # TODO using mkldnn weight cache

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
            elif args.mkldnn:
                inputs["video"] = inputs["video"].to_mkldnn()
                inputs["audio"] = inputs["audio"].to_mkldnn()

            output = model(inputs)

            if args.mkldnn:
                output = output.to_dense()

            loss_data = loss(output, target)
            # TODO get accuracy
            #print(meters)
            #meters.update(output, target)
            #dic = meters.get_classy_state()
            #print(dic)
            batch_time.update(_time(args.cuda) - end)
            end = _time(args.cuda)

            if i % args.print_freq == 0:
                progress.display(i)
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

