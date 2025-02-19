import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch.optim import AdamW, lr_scheduler
import numpy as np
from tqdm import tqdm
import time
import argparse
from logger import Logger
import json
from collections import namedtuple
import h5py

from torch.cuda.amp import autocast as autocast, GradScaler
from accelerate import Accelerator

from datasets import get_dataset
from opensora import get_conditional_dit
from loss import EDMLoss, LossArgs
from utils import log_stats

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='EDM for kol')
parser.add_argument('--seed', type=int,                             default=42,                     help='random seed')
parser.add_argument('--data', type=str,                             default='kol',                  help='data_name: kol')
parser.add_argument('--version', type=str,                          default='latent',                   help='data_name: kol')
parser.add_argument('--data_loc', type=str,                         default="/media/group3/lzy/Data/kol/kf_2d_re1000_latent_40seed_bthwc.npy",          help='path of dataset')
parser.add_argument('--results_path', type=str,                     default='__results__',          help='root path for saving results')
parser.add_argument('--ckpt_name', type=str,                        default='ckpt.pt',              help='name of checkpoint')
parser.add_argument('--continue_training', type=int,                default=0,                      help='data_name: kol')
parser.add_argument('--print_freq', type=int,                       default=10,                      help='frequency of info. printing')

parser.add_argument('--train_portion', type=float,                  default=1.0,                    help='data split')
parser.add_argument('--crop_size', type=int,                        default=32,                     help='patched training')
parser.add_argument('--num_train', type=int,                        default=100000000,              help='number of samples for training')
parser.add_argument('--num_val', type=int,                          default=100,                    help='number of samples for validation')
parser.add_argument('--num_frames', type=int,                       default=16,                     help='number of frames')
parser.add_argument('--sep_t_channel', type=int,                    default=1,                      help='whether to output \Sigma')

parser.add_argument('--model_name', type=str,                       default='diffusion',            help='model name')
parser.add_argument('--input_size', type=int, nargs='+',            default=(16, 32, 32),           help='latent size')
parser.add_argument('--in_channels', type=int,                      default=4)
parser.add_argument('--patch_size', type=int, nargs='+',            default=(1, 2, 2),              help='patch size T*H*W')
parser.add_argument('--hidden_size', type=int,                      default=512,                    help='hidden size')
parser.add_argument('--depth', type=int,                            default=8,                      help='number of modules')
parser.add_argument('--num_heads', type=int,                        default=16,                     help='number of attention heads')
parser.add_argument('--mlp_ratio', type=float,                      default=1,                      help='mlp ratio')
parser.add_argument('--learn_sigma', type=int,                      default=0,                      help='whether to output \Sigma')
parser.add_argument('--dtype', type=str,                            default='float16',              help='data type')
parser.add_argument('--condition', type=str,                        default='label_2',              help='condition type')
parser.add_argument('--class_dropout_prob', type=float,             default=0.1,                    help='condition dropout prob.')
parser.add_argument('--cross_attn', type=int,                       default=0,                      help='whether to use cross-attention for additional conditioning info.')
parser.add_argument('--frame_cond_strategy', type=str,              default='none',                 help='whether to condition on some frames: none, fixed_2')
parser.add_argument('--enable_flash_attn', type=int,                default=1,                      help='whether to enable flash attention')
parser.add_argument('--enable_layernorm_kernel', type=int,          default=0,                      help='whether to enable apex')


parser.add_argument('--batch_size', type=int,                       default=64)
parser.add_argument('--small_batch_size', type=int,                 default=64)
parser.add_argument('--n_iters', type=int,                          default=100000,                 help='total number of iterations')
parser.add_argument('--lr', type=float,                             default=1e-4,                   help='learning rate')
parser.add_argument('--ema_rate', type=float,                       default=0.999,                  help='ema rate')

parser.add_argument('--is_scalar', type=int,                        default=0,                      help='whether to normalize data')
parser.add_argument('--sigma_data', type=float,                     default=0.696,                  help='std. of the dataset')
parser.add_argument('--P_mean', type=float,                         default=-1.2,                   help='P_mean for noise sampling')
parser.add_argument('--P_std', type=float,                          default=1.2,                    help='P_std  for noise sampling')
config = parser.parse_args()


def train(opt, log):
    batch_size = opt.batch_size
    lr = opt.lr
    ema_rate = opt.ema_rate
    sigma_data = opt.sigma_data
    P_mean = opt.P_mean
    P_std = opt.P_std
    checkpoint_save_path = os.path.join(opt.results_path, opt.ckpt_name)
    tb_save_path = opt.results_path
    accelerator = Accelerator(mixed_precision='fp16')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = accelerator.device
    dtype = getattr(torch, opt.dtype)

    net = get_conditional_dit(opt)
    ema = ExponentialMovingAverage(net.parameters(), decay=ema_rate)
    loss_func = EDMLoss(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)
    optimizer = AdamW(net.parameters(), lr=lr, weight_decay=0, eps=1e-8)
    sched = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
    get_loss_args = LossArgs(opt)
    # scalar = GradScaler()
    writer = SummaryWriter(log_dir=tb_save_path)
    log.info(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")

    start_iter = 0
    if opt.continue_training:
        ckpt = torch.load(checkpoint_save_path)
        net.load_state_dict(ckpt['net'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['sched'])
        start_iter = int(ckpt['optimizer']['state'][0]['step'])
        log.info(f"Checkpoint loaded from iter no.: {start_iter}")

    net = nn.DataParallel(net)
    # net.to(opt.device)
    ema.to(device)

    if '.npy' in opt.data_loc:
        data = np.load(opt.data_loc)
    elif '.h5' in opt.data_loc:
        data = {'xs': h5py.File(opt.data_loc, 'r')['xs'][:], 'ys': h5py.File(opt.data_loc, 'r')['ys'][:]}
    else:
        data = opt.data_loc
    dataset = get_dataset(opt, log, data, train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    net, optimizer, dataloader, sched = accelerator.prepare(net, optimizer, dataloader, sched)

    net.train()
    i_outer = start_iter
    i_inner = 0
    n_inner_loop = opt.batch_size // opt.small_batch_size
    for i, (x0, y) in enumerate(dataloader):      # x0 refers to the high-res label data, while y can be any thing that you want to use in corrupt_func
        if i % n_inner_loop == 0:
            optimizer.zero_grad()
        
        # x0 = x0.to(dtype)
        y = y.float().to(device) if y.any() else y.long()

        loss_args = get_loss_args(x0)
        loss = loss_func(net, x0, y, square=True, **loss_args).mean()
        # print(loss.item())
        accelerator.backward(loss)
        i_inner += 1

        if i_inner == n_inner_loop:
            optimizer.step()
            ema.update()
            sched.step()
            # scalar.update()
            i_outer += 1
            i_inner = 0
            if i_outer == opt.n_iters:
                break
        
            '''logging'''
            if i_outer % opt.print_freq == 0:
                lr_curr = optimizer.param_groups[0]['lr']
                log.info(f'training {i_outer}/{opt.n_iters} | lr: {lr_curr:.2e} | loss: {loss.item():.4f}')
                stats = {
                    'iterations': i_outer,
                    'lr': lr_curr,
                    'loss': loss.item()
                }
                log_stats(stats, writer, i_outer)
            
            '''save model'''
            if i_outer % 500 == 0:
                torch.save({
                    "net": net.module.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict() if sched is not None else sched,
                }, checkpoint_save_path)
                log.info(f"Saved latest({i_outer=}) checkpoint to {checkpoint_save_path=}!")
            
            if i_outer % 10000 == 0:
                torch.save({
                    "net": net.module.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict() if sched is not None else sched,
                }, opt.results_path + f'/checkpoint_{i_outer}.pt')
    return


if __name__ == "__main__":

    '''create results folder'''
    path = config.results_path + '/' + config.data + '_' + config.version
    config.results_path = path

    used_para = dict(
        batch_size=config.batch_size,
        small_batch_size=config.small_batch_size,
        )

    if not os.path.exists(path):
        os.mkdir(path)
    if not config.continue_training:
        with open(config.results_path + "/opt.json", mode="w") as f:
            json.dump(config.__dict__, f, indent=4)
    else:
        '''load option file'''
        opt_path = path + '/opt.json'
        with open(opt_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config['continue_training'] = True
            for key in used_para.keys():
                config[key] = used_para[key]
        OPT_class = namedtuple('OPT_class', config.keys())
        config = OPT_class(**config)

    log = Logger(0, path)
    log.info('**************************************')
    log.info('           start training !           ')
    log.info('**************************************')
    train(config, log)
