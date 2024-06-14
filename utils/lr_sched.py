# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(config, iteration):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if iteration < config.warmup_epochs:
        lr = config.learning_rate * iteration / config.warmup_epochs
    else:
        lr = config.minimum_learning_rate + (config.learning_rate - config.minimum_learning_rate) * 0.5 * \
            (1. + math.cos(math.pi * (iteration - config.warmup_epochs) / (config.num_epochs - config.warmup_epochs)))
    for param_group in config.optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
