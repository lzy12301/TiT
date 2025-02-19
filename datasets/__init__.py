from .kol import DatasetKol
import torch
import numpy as np


def get_dataset(opt, log, data, train=True):
    kwags = dict(opt=opt, log=log, data=data, train=train)
    if 'kol' in opt.data.lower():
        return DatasetKol(**kwags)
    else:
        raise ValueError('Unknown type of dataset!')
    return 


def get_bound_func(bound_type='periodic', num_grid=3):
    if 'periodic' in bound_type.lower():
        return lambda x: bound_func_periodic(x, num_grid)
    else:
        raise ValueError('Unknown type of boundary condition!')


def bound_func_periodic(field, num_grid=3):
    roll_fn = torch.roll if isinstance(field, torch.Tensor) else np.roll
    cat_fn = torch.cat if isinstance(field, torch.Tensor) else np.concatenate
    if isinstance(field, torch.Tensor):
        field_ = cat_fn([roll_fn(field, num_grid, -2)[..., :num_grid, :].clone().detach(), field, roll_fn(field, -num_grid, -2)[..., -num_grid:, :].clone().detach()], -2)
        field_ = cat_fn([roll_fn(field_, num_grid, -1)[..., :num_grid].clone().detach(), field_, roll_fn(field_, -num_grid, -1)[..., -num_grid:].clone().detach()], -1)
    else:
        field_ = cat_fn([roll_fn(field, num_grid, -2)[..., :num_grid, :], field, roll_fn(field, -num_grid, -2)[..., -num_grid:, :]], -2)
        field_ = cat_fn([roll_fn(field_, num_grid, -1)[..., :num_grid], field_, roll_fn(field_, -num_grid, -1)[..., -num_grid:]], -1)
    return field_
