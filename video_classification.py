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
import time
import os
import model_config
import argparse


parser = argparse.ArgumentParser(description='PyTorch Video UCF101 Training')
parser.add_argument('video_dir', metavar='DIR',
                    help='path to video files')
parser.add_argument('--num_epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')

def main():
    args = parser.parse_args()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        print("Using GPU device to do this work")

    # set it to the folder where video files are saved
    video_dir = args.video_dir + "/UCF-101"
    # set it to the folder where dataset splitting files are saved
    splits_dir = args.video_dir + "/ucfTrainTestlist"
    # set it to the file path for saving the metadata
    metadata_file = args.video_dir + "/metadata.pth"

    resnext3d_configs =model_config.ResNeXt3D_Config(video_dir, splits_dir, metadata_file, args.num_epochs)
    resnext3d_configs.setUp()

    datasets = {}
    datasets["train"] = build_dataset(resnext3d_configs.dataset_configs["train"])
    datasets["test"] = build_dataset(resnext3d_configs.dataset_configs["test"])

    model = build_model(resnext3d_configs.model_configs)
    meters = build_meters(resnext3d_configs.meters_configs)
    loss = build_loss({"name": "CrossEntropyLoss"})

    #print(model)
    if args.evaluate:
        '''
        # This can run eval, but can not get runing time for given iteration
        # so we maually runing the forward step
        task.prepare(use_gpu=args.cuda)
        task.advance_phase() # will get train step
        task.advance_phase() # will get test step
        local_variables = {}

        task.eval_step(use_gpu = args.cuda, local_variables = local_variables)
        '''

        print("Running evaluation step")
        iterator = datasets["test"].iterator(shuffle_seed = 0,
                                             epoch= 0,
                                             num_workers=0,  # 0 indicates to do dataloading on the master process
                                             pin_memory=False,
                                             multiprocessing_context=None)

        model = model.eval()
        if args.cuda:
            model = model.cuda()
        for sample in iter(iterator):
            inputs = sample["input"]
            target = sample["target"]
            if args.cuda:
               inputs["video"] = inputs["video"].cuda()
               inputs["audio"] = inputs["audio"].cuda()
               target = target.cuda()
            with torch.no_grad():
                output = model(inputs)
                loss_data = loss(output, target)
                print("running....")
                #print(meters)
                #meters.update(output, target)
                #dic = meters.get_classy_state()
                #print(dic)
    else:
        print("Running training step")
        optimizer = build_optimizer(resnext3d_configs.optimizer_configs)

        task = (
            ClassificationTask()
            .set_num_epochs(args.num_epochs)
            .set_loss(loss)
            .set_model(model)
            .set_optimizer(optimizer)
            .set_meters(meters)
        )
        for phase in ["train", "test"]:
            task.set_dataset(datasets[phase], phase)

        hooks = [LossLrMeterLoggingHook(log_freq=4)]

        checkpoint_dir = f"/home/xiaobinz/Downloads/resnext3d/ucf/checkpoint/classy_checkpoint_{time.time()}"
        os.mkdir(checkpoint_dir)
        hooks.append(CheckpointHook(checkpoint_dir, input_args={}))

        task = task.set_hooks(hooks)
        trainer = LocalTrainer(use_gpu = args.cuda)
        trainer.train(task)

if __name__ == '__main__':
    main()

