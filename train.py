import os
import sys
import math
import time
import argparse
import datetime
import json
from pathlib import Path
from typing import Iterable

import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import webdataset as wds
import torch
print(torch.__version__)
import timm.optim.optim_factory as optim_factory

import misc
from models import TwoCropsTransform, SimSiam, vimlp_huge
from misc import NativeScalerWithGradNormCount as NativeScaler


GLOBAL_ITER = 0


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Batch size per GPU (effective batch size is batch_size_per_gpu * accum_iter * # gpus')
    parser.add_argument('--epochs', default=999, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_prefix', default='', type=str, help='prefix for saving checkpoint and log files')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, help='learning rate (absolute lr)')
    parser.add_argument('--data_path', default='/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    parser.add_argument('--saveckp_freq', default=10000, type=int, help='Save checkpoint every x iterations.')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def preprocess(sample):
    return sample[0]


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    cudnn.benchmark = True

    # simple augmentations
    transform = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    transform = TwoCropsTransform(transforms.Compose(transform))

    # use webdataset for loading data
    dataset = (wds.WebDataset(args.data_path, resampled=True).shuffle(10000, initial=10000).decode("pil").to_tuple("jpg").map(preprocess).map(transform))
    data_loader = wds.WebLoader(dataset, shuffle=False, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers)
    
    # define the model
    model = SimSiam(vimlp_huge, args.dim, args.pred_dim)
    model.to(device)

    # effective batch size
    eff_batch_size = args.batch_size_per_gpu * args.accum_iter * misc.get_world_size()
    print("effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module
    
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CosineSimilarity(dim=1).cuda(args.gpu)

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    optimizer.lr = args.lr  # override loaded lr
    
    print("Starting SimSiam training!")
    start_time = time.time()
    for _ in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, data_loader, optimizer, criterion, device, loss_scaler, args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device, loss_scaler, args=None):
    
    global GLOBAL_ITER

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(data_loader):

        samples[0] = samples[0].to(device, non_blocking=True)
        samples[1] = samples[1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            p1, p2, z1, z2 = model(x1=samples[0], x2=samples[1])
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if GLOBAL_ITER % args.saveckp_freq == 0:
            # ============ writing logs + saving checkpoint ============
            save_dict = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'it': GLOBAL_ITER,
                'args': args,
                'scaler': loss_scaler.state_dict(),
            }

            misc.save_on_master(save_dict, os.path.join(args.output_dir, args.save_prefix + '_checkpoint.pth'))

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'it': GLOBAL_ITER}

            if misc.is_main_process():
                with (Path(args.output_dir) / (args.save_prefix + "_log.txt")).open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # start a fresh logger to wipe off old stats
            metric_logger = misc.MetricLogger(delimiter="  ")

        GLOBAL_ITER += 1

    print('Out of epoch loop')

    return train_stats


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)