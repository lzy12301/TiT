import torch
import torch.nn.functional as F
import torch.nn as nn


class VICReg(nn.Module):
    def __init__(
            self,
            sim_coeff: float,
            cov_coeff: float,
            std_coeff: float,
            batch_size: int,
            mlp: str,
            n_time_steps: int,
            ):
        super().__init__()
        self.num_features = int(mlp.split("-")[-1])
        import torchvision.models.resnet as resnet
        self.backbone = resnet.__dict__['resnet18'](pretrained=False)
        self.backbone.fc = nn.Identity()
        # Change number of channels
        self.backbone.conv1 = torch.nn.Conv2d(
            n_time_steps * 5,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias
            )
        _, out_dim = self.backbone(torch.zeros(1, n_time_steps*5, 224, 224)).shape

        self.projector = Projector(out_dim, mlp)
        self.batch_size = batch_size
        self.sim_coeff = sim_coeff
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff

    def forward(self, x, y):
        # we do not apply inverse transform for now
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


def Projector(embedding, dimensions):
    mlp_spec = f"{embedding}-{dimensions}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def RegressionHead(embedding, dimensions, sig_max=0.1, sig_min=0.6):
    mlp_spec = f"{embedding}-{dimensions}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    # we add a sigmoid to constrain the value in a range,
    # making it easier for the classifier to be calibrated
    layers.append(RangeSigmoid(max=sig_max, min=sig_min))
    return nn.Sequential(*layers)


class ResNetBaseline(nn.Module):
    def __init__(self, target_dim, n_time_steps) -> None:
        super().__init__()
        import torchvision.models.resnet as resnet
        self.backbone = resnet.__dict__['resnet18'](pretrained=False)
        self.backbone.fc = nn.Identity()

        self.backbone.conv1 = torch.nn.Conv2d(
            n_time_steps * 5, self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias,
            )
        _, out_dim = self.backbone(torch.zeros(1, n_time_steps * 5, 224, 224)).shape
        self.regression_head = RegressionHead(out_dim, target_dim)

    def forward(self, input):
        output = self.backbone(input)
        return self.regression_head(output)
    
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


# ========================================================
#                Random utils functions
# ========================================================
import torchvision
import json
import time

def log_stats(stats, writer, epoch):
    for k, v in stats.items():
        writer.add_scalar(k, v, epoch)


def log_imgs(imgs_to_log, writer, epoch):
    legend, imgs = imgs_to_log
    imgs = [torch.unsqueeze(img.detach(), dim=0) for img in imgs]
    grid = torchvision.utils.make_grid(imgs, nrow=1, normalize=True, scale_each=True, padding=2, pad_value=1.0)
    writer.add_image(legend, grid, epoch)


class RangeSigmoid(nn.Module):
    def __init__(self, max, min):
        super().__init__()
        self.max = max
        self.min = min

    def forward(self, input):
        return torch.sigmoid(input) * (self.max - self.min) + self.min


def log(folder, content, start_time):
        print(f'=> Log: {content}')
        # if self.rank != 0: return
        cur_time = time.time()
        with open(folder + '/log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - start_time,
                **content
            }) + '\n')
            fd.flush()


def relative_error(y_pred, y):
    err = torch.abs(y_pred - y) / torch.abs(y)
    return torch.mean(err)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
