#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

# why we need this? mkldnn not support permute and mean now

class MkldnnFullyConvolutionalLinear(nn.Module):
    def __init__(self, dim_in, num_classes, act_func="softmax"):
        super(MkldnnFullyConvolutionalLinear, self).__init__()
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.to_dense()
        x = x.permute((0, 2, 3, 4, 1))
        x = self.projection(x)
        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)
        return x
