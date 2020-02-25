import torch
import copy

class ResNeXt3D_Config:
    def __init__(self, video_dir, splits_dir, metadata_file, num_epochs):
        self.video_dir = video_dir
        self.splits_dir = splits_dir
        self.metadata_file = metadata_file
        self.num_epochs = num_epochs

    def setUp(self):
        self.model_configs = {
            "name": "resnext3d",
            "frames_per_clip": 32,
            "input_planes": 3,
            "clip_crop_size": 112,
            "skip_transformation_type": "postactivated_shortcut",
            #"residual_transformation_type": "basic_transformation",
            "residual_transformation_type": "postactivated_bottleneck_transformation",
            "num_blocks": [3, 4, 23, 3],
            "input_key": "video",
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 3,
            "stem_maxpool": False,
            "stage_planes": 64,
            #"stage_temporal_kernel_basis": [[3], [3], [3], [3]],
            "stage_temporal_kernel_basis": [[1],[1], [1], [1]],
            #"temporal_conv_1x1": [False, False, False, False],
            "temporal_conv_1x1": [True, True, True, True],
            #"stage_temporal_stride": [1, 2, 2, 2],
            "stage_temporal_stride": [1, 1, 1, 1],
            "stage_spatial_stride": [1, 2, 2, 2],
            "num_groups": 32,
            "width_per_group":8,
            "num_classes": 101,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                     "unique_id": "default_head",
                     "pool_size": [4, 7, 7],
                     "activation_func": "softmax",
                     "num_classes": 101,
                     "fork_block": "pathway0-stage4-block2",
                     "in_plane": 512
                }
            ]
        }

        self.dataset_configs = {
            "train":{
            "name": "ucf101",
                "split": "train",
                "batchsize_per_replica": 5,
                "use_shuffle": True,
                "num_samples": None,
                "frames_per_clip": 32,
                "step_between_clips": 1,
                "clips_per_video": 1,
                "video_dir": self.video_dir,
                "splits_dir": self.splits_dir,
                "metadata_file": self.metadata_file,
                "fold": 1,
                "transforms": {
                    "video": [
                {
                            "name": "video_default_augment",
                             "crop_size": 112,
                             "size_range": [128, 160]
                        }
            ]
                }
            },
        "test": {
                "name": "ucf101",
                "split": "test",
                "batchsize_per_replica": 10,
                "use_shuffle": False,
                "num_samples": None,
                "frames_per_clip": 32,
                "step_between_clips": 1,
                "clips_per_video": 10,
                "video_dir": self.video_dir,
                "splits_dir": self.splits_dir,
                "metadata_file": self.metadata_file,
                "fold": 1,
                "transforms": {
                    "video": [
                        {
                            "name": "video_default_no_augment",
                            "size": 128
                        }
                    ]
                }
            }
        }

        self.meters_configs = {
            "accuracy": {
                "topk": [1, 5]
            },
            "video_accuracy": {
                "topk": [1, 5],
                "clips_per_video_train": 1,
                "clips_per_video_test": 10
            }
        }

        self.optimizer_configs = {
            "name": "sgd",
            "param_schedulers": {
            "lr": {
                 "name": "composite",
                 "schedulers": [
                     {
                         "name": "linear",
                         "start_lr": 0.005,
                         "end_lr": 0.04
                     },
                     {
                         "name": "cosine",
                         "start_lr": 0.04,
                         "end_lr": 0.00004
                     }
                 ],
                 "lengths": [0.13, 0.87],
                 "update_interval": "epoch",
                 "interval_scaling": ["rescaled", "rescaled"]
                 }
             },
             "num_epochs": self.num_epochs,
             "weight_decay": 0.005,
             "momentum": 0.9,
             "nesterov": True
         }

