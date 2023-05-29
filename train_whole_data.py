# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 10:56
# @Author  : Falcon
# @FileName: train_whole_data.py.py
import os
import argparse
import sys
import datetime
import numpy as np
import math
from typing import Iterable, Optional
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import json

from pathlib import Path

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy

from data.get_dataset import build_dataset
from utils.samplers import RASampler
from models.modeling_timm import get_vit_base_patch16_224, get_vit_large_patch16_224, get_vit_tiny_patch16_224
from models.t2t_vit import get_t2t_vit_14, get_t2t_vit_t_14, get_t2t_vit_7

from utils import dist_utils
from utils.dist_utils import get_rank, get_world_size, init_distributed_mode
from utils.logger import create_logger


def get_args_parser():
    parser = argparse.ArgumentParser('ViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--eval-batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=5, type=int)

    # Model parameters
    # parser.add_argument('--model', default='cct7_7x2', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    # parser.add_argument('--model', default='twins_pcpvt_small', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    # parser.add_argument('--model-path', type=str,
    #                     default=r'')
    # parser.add_argument('--model', default='twins_svt_small', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    # parser.add_argument('--model-path', type=str,
    #                     default=r'F:\program_lab\python\py3\decomposition_cv\checkpoint\twins\twins_svt_small-42e5f78c.pth')
    # parser.add_argument('--model', default='t2t_vit_t_14', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    # parser.add_argument('--model-path', type=str,
    #                     default=r'F:\program_lab\python\py3\decomposition_cv\checkpoint\t2t\81.7_T2T_ViTt_14.pth.tar')
    # parser.add_argument('--model', default='vit_large', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    # parser.add_argument('--model-path', type=str,
    #                     default=r'F:\program_lab\python\py3\decomposition_cv\checkpoint\vit\pretrain_im1k\L_16-i1k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz')
    parser.add_argument('--model', default='vit_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model-path', type=str,
                        default=r'F:\program_lab\python\py3\decomposition_cv\checkpoint\vit\others\B_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    parser.add_argument('--no-aug', action='store_true', help='not use aug')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default=r'F:\program_lab\python\dataset',
                        type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='cifar100',
                        choices=['cifar100', 'flowers', 'cars', 'pets', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num_division', metavar='N',
                        type=int,
                        default=4,
                        help='The number of sub models')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default=r'F:\program_lab\python\py3\decomposition_cv\output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    # parser.add_argument('--eval', type=bool, default=True, help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--disable_amp', action='store_true', default=False,
                        help='disable AMP')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # args = parser.parse_args()
    return parser


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None):
    model.train(mode=True)
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.disable_amp:
            outputs = model(samples)
            loss = criterion(outputs, targets)
        else:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            log.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        # if model_ema is not None:
        #     model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # log.info('Top-1: {top1.global_avg:.3f} \t Top5: {top5.global_avg:.3f} \t loss: {losses.global_avg:.3f}'
    #          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    init_distributed_mode(args)

    # log init
    logger = create_logger(output_dir=args.output_dir, dist_rank=get_rank(), name=f"{args.method}")

    logger.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_dataset, test_dataset, num_classes = build_dataset(args)
    args.num_classes = num_classes

    num_tasks = get_world_size()
    global_rank = get_rank()

    if args.repeated_aug:
        sampler_train = RASampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    if args.dist_eval:
        if len(test_dataset) % num_tasks != 0:
            logger.info(
                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        test_dataset, sampler=sampler_val,
        # batch_size=int(1.5 * args.batch_size),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    logger.info(f"Creating model: {args.model}")
    model = None
    if args.model == 'vit_large':
        model = get_vit_large_patch16_224(pretrained_path=args.model_path, num_classes=num_classes)
    elif args.model == 'vit_tiny':
        model = get_vit_tiny_patch16_224(pretrained_path=args.model_path, num_classes=num_classes)
    elif args.model == 'vit_base':
        model = get_vit_base_patch16_224(pretrained_path=args.model_path, num_classes=num_classes)
    elif args.model == 't2t_vit_t_14':
        model = get_t2t_vit_t_14(pretrained_path=args.model_path, num_classes=num_classes)
    elif args.model == 't2t_vit_14':
        model = get_t2t_vit_14(pretrained_path=args.model_path, num_classes=num_classes)
    elif args.model == 'twins_pcpvt_small':
        from models.twins import get_twins_pcpvt_small
        model = get_twins_pcpvt_small(pretrained_path=args.model_path, num_classes=num_classes)
    elif args.model == 'twins_svt_small':
        from models.twins import get_twins_svt_small
        model = get_twins_svt_small(pretrained_path=args.model_path, num_classes=num_classes)
    else:
        if args.model.split('_')[0] == 'cct2':
            from models.cct import get_cct2
            cfg = args.model.split('_')[-1]
            kernel_size, conv_layers = [int(i) for i in cfg.split('x')]
            model = get_cct2(img_size=args.input_size, kernel_size=kernel_size,
                             n_conv_layers=conv_layers)
        elif args.model.split('_')[0] == 'cct6':
            from models.cct import get_cct6
            cfg = args.model.split('_')[-1]
            kernel_size, conv_layers = [int(i) for i in cfg.split('x')]
            model = get_cct6(img_size=args.input_size, kernel_size=kernel_size,
                             n_conv_layers=conv_layers)
        elif args.model.split('_')[0] == 'cct7':
            from models.cct import get_cct7
            cfg = args.model.split('_')[-1]
            kernel_size, conv_layers = [int(i) for i in cfg.split('x')]
            model = get_cct7(img_size=args.input_size, kernel_size=kernel_size,
                             n_conv_layers=conv_layers)
        elif args.model.split('_')[0] == 'cct14':
            from models.cct import get_cct14
            cfg = args.model.split('_')[-1]
            kernel_size, conv_layers = [int(i) for i in cfg.split('x')]
            model = get_cct14(img_size=args.input_size, kernel_size=kernel_size,
                              n_conv_layers=conv_layers)

        if args.model_path != '':
            load_dict = torch.load(args.model_path)
            model.load_state_dict(load_dict)
            model.reset_classifier(num_classes=num_classes)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()
    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%")
        return

    logger.info(f"Start training")
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # init tensorboard
    writer = SummaryWriter(log_dir=output_dir) if get_rank() == 0 else None

    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model=model, criterion=criterion, data_loader=data_loader_train,
                                      optimizer=optimizer,
                                      device=device, epoch=epoch, loss_scaler=loss_scaler, log=logger,
                                      max_norm=args.clip_grad,
                                      mixup_fn=mixup_fn)

        lr_scheduler.step(epoch)
        if output_dir and dist_utils.is_main_process():
            checkpoint_path = os.path.join(output_dir, 'checkpoint_temp.pth')
            dist_utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
        if writer is not None:
            writer.add_scalar('Train/loss', train_stats['loss'], epoch)
            writer.add_scalar('Train/lr', train_stats['lr'], epoch)
        logger.info(f"Epoch: {epoch}/{args.epochs} \t [Train] Loss: {train_stats['loss']:.4f} \t ")

        test_stats = evaluate(data_loader=data_loader_val, model=model, device=device)
        if writer is not None:
            writer.add_scalar('Test/loss', test_stats['loss'], epoch)
            writer.add_scalar('Test/Top1', test_stats['acc1'], epoch)
            writer.add_scalar('Test/Top5', test_stats['acc5'], epoch)
        logger.info(f"Epoch: {epoch}/{args.epochs} \t [Eval] Top-1: {test_stats['acc1']:.4f} \t "
                    f"Top-5: {test_stats['acc5']:.4f} \t Loss: {test_stats['loss']:.4f} \t ")

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir and dist_utils.is_main_process():
                model_checkpoint = os.path.join(output_dir, f"checkpoint.pth")
                torch.save(model_without_ddp.state_dict(), model_checkpoint)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                with open(os.path.join(output_dir, 'result.txt'), 'w') as f:
                    f.write(f'Final Accuracy: {max_accuracy}')
                logger.info(f'Saving model in [PATH]: {output_dir}')

        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and dist_utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f'Epochs: {epoch} \t Training time: {total_time_str} ')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time: {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('vit training and evaluation script on whole dataset', parents=[get_args_parser()])
    args = parser.parse_args()
    args.name = f'{args.model}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-grad{args.clip_grad}' \
                f'-wd{args.weight_decay}-no_aug' if args.no_aug else f'{args.model}-lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-grad{args.clip_grad}' \
                f'-wd{args.weight_decay}'
    args.method = f'train_whole_timm'
    args.output_dir = os.path.join(args.output_dir, 'vit_timm', args.data_set, args.method, args.model, args.name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

