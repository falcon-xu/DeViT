# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 15:43
# @Author  : Falcon
# @FileName: distill_sub.py
import os
import json
import argparse
import datetime
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma, get_state_dict
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from engine import evaluate, train_1epoch_qkv
from data.get_dataset import build_division_dataset

import models.de_vit
from models.de_vit import model_config
from core.imp_rank import *
from utils import dist_utils
from utils.samplers import RASampler
from utils.logger import create_logger
from utils.losses import DistillLoss
from utils.dist_utils import get_rank, get_world_size


def get_args_parser():
    parser = argparse.ArgumentParser('ViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--eval-batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--output_dir', default=r'./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune")

    # Model parameters
    parser.add_argument('--model', default='dedeit', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model-path', type=str, default='')
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
    parser.add_argument('--no_aug', action='store_true', help="no aug")

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

    # Distillation parameters
    parser.add_argument('--teacher-model', default='vit_large_patch16_224', type=str, metavar='MODEL',
                        help='Name of teacher model to train')
    parser.add_argument('--teacher-path', type=str,
                        default=r'./teacher_ckpt')
    parser.add_argument('--distillation-type', default='hard', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-inter', type=bool, default=True,
                        help="Whether to distill intermediate features")
    # parser.add_argument('--distillation-token', type=bool, default=True, help="Whether to distill token")
    parser.add_argument('--distillation-token', action='store_true', help="Whether to distill token")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    # Loss parameters
    parser.add_argument('--gama', nargs='+', default=[0.2, 0.1, 0.3], help="paras to adjust loss")

    # Dataset parameters
    parser.add_argument('--data-path', default=r'./datasets',
                        type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='cifar100', choices=['cifar100', 'IMNET', 'cars', 'pets', 'flowers'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    # Division parameters
    parser.add_argument('--num_division', metavar='N',
                        type=int,
                        default=4,
                        help='The number of sub models')
    parser.add_argument('--start-division', metavar='N',
                        type=int,
                        default=0,
                        help='The number of sub models')

    # Others
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # shrink
    parser.add_argument('--load_shrink', action='store_true', default=False)
    parser.add_argument('--shrink_checkpoint', type=str, default='')
    parser.add_argument('--neuron_shrinking', action='store_true', default=False)
    parser.add_argument('--head_shrinking', action='store_true', default=False)

    # args = parser.parse_args()
    return parser


def get_models(args, num_classes, num_sub, log):
    stu_nb = 1000 if args.model_path != '' else num_classes
    model_path = args.model_path if args.finetune else None
    resize_dim = model_config[args.teacher_model]["embed_dim"] if args.distillation_token else None
    model = create_model(
        args.model,
        pretrained=True,
        pretrained_path=model_path,
        num_classes=stu_nb,
        resize_dim=resize_dim,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    log.info(f'Create {args.model} model\n Load ckpt from [PATH]: {model_path}')
    if args.model_path != '':
        model.reset_classifier(num_classes=num_classes)
    model.to(args.device)

    teacher_model = None
    if args.distillation_type != 'none':
        teacher_path = os.path.join(args.teacher_path, f'sub-dataset{num_sub}', 'checkpoint.pth')
        teacher_ckpt = torch.load(teacher_path, map_location='cpu')
        teacher_model = create_model(args.teacher_model,
                                     num_classes=num_classes,
                                     drop_rate=args.drop,
                                     drop_path_rate=args.drop_path,
                                     drop_block_rate=None, )
        teacher_model.load_state_dict(teacher_ckpt)
        teacher_model.to(args.device)
        teacher_model.eval()

    return model, teacher_model


def main(args):
    dist_utils.init_distributed_mode(args)

    # Create output path
    args.method = f'distill_sub'
    args.name = f'lr{args.lr}-bs{args.batch_size}-epochs{args.epochs}-grad{args.clip_grad}' \
                f'-wd{args.weight_decay}-wm{args.warmup_epochs}-gama{args.gama[0]}_{args.gama[1]}_{args.gama[2]}'
    args.output_dir = os.path.join(args.output_dir, f'{args.dataset}_div{args.num_division}', f'{args.model}',
                                   args.method, args.name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # log init
    logger = create_logger(output_dir=args.output_dir, dist_rank=dist_utils.get_rank(), name=f"{args.method}")
    logger.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + dist_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Load dataset
    sub_dataset_path = os.path.join(args.data_path, f'sub-dataset{args.start_division}')

    train_dataset, test_dataset, division_num_classes = build_division_dataset(dataset_path=sub_dataset_path, args=args)
    args.num_classes = division_num_classes

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
            label_smoothing=args.smoothing, num_classes=division_num_classes)

    model, teacher_model = get_models(args, division_num_classes, args.start_division, logger)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    linear_scaled_lr = args.lr * args.batch_size * get_world_size() / 512.0
    args.lr = linear_scaled_lr
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

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillLoss(base_criterion=criterion, distillation_type=args.distillation_type,
                            alpha=args.distillation_alpha, tau=args.distillation_tau)

    if args.resume:
        logger.info(f'Load resume checkpoint from [PATH]: {args.resume}')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and not args.finetune and \
                'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                dist_utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, args.device)
        logger.info(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%")
        return

    # load shrink
    if args.shrink_checkpoint != '':
        shrinked_policy = np.load(os.path.join(args.shrink_checkpoint, 'shrinked_policy.npy'))
        shrinked_acc = np.load(os.path.join(args.shrink_checkpoint, 'shrinked_accuracy.npy'))
        max_index = np.argmax(shrinked_acc)
        neuron_sparsity = shrinked_policy[max_index, :12]
        head_sparsity = shrinked_policy[max_index, 12:-1]

    if args.neuron_shrinking:
        logger.info('Start shrink neuron')
        neuron_mask = mlp_neuron_mask(model_without_ddp, neuron_sparsity,
                                      mlp_neuron_rank(model_without_ddp, data_loader_train))
        mlp_neuron_shrink(model_without_ddp, neuron_mask)

    if args.head_shrinking:
        logger.info('Start shrink head')
        head_mask = attn_head_mask(model_without_ddp, head_sparsity,
                                   attn_head_rank(model_without_ddp, data_loader_train))
        attn_head_shrink(model_without_ddp, head_mask)

    logger.info(f"Start training for {args.epochs} epochs in sub-dataset{args.start_division}")
    output_dir = Path(os.path.join(args.output_dir, f'sub-dataset{args.start_division}'))
    os.makedirs(output_dir, exist_ok=True)

    # init tensorboard
    writer = SummaryWriter(log_dir=output_dir) if get_rank() == 0 else None

    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_1epoch_qkv(model=model, teacher_model=teacher_model, criterion=criterion, args=args,
                                       data_loader=data_loader_train, optimizer=optimizer, device=args.device,
                                       epoch=epoch, loss_scaler=loss_scaler, log=logger, max_norm=args.clip_grad,
                                       model_ema=model_ema, mixup_fn=mixup_fn)

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_temp.pth']
            for checkpoint_path in checkpoint_paths:
                dist_utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader=data_loader_val, model=model, device=args.device)
        logger.info(f"Epoch: {epoch}/{args.epochs} \t [Train] Loss: {train_stats['loss']:.4f} \t ")
        logger.info(f"Epoch: {epoch}/{args.epochs} \t [Eval] Top-1: {test_stats['acc1']:.4f} \t "
                    f"Top-5: {test_stats['acc5']:.4f} \t Loss: {test_stats['loss']:.4f} \t ")
        if writer is not None:
            writer.add_scalar('Train/loss', train_stats['loss'], epoch)
            writer.add_scalar('Train/lr', train_stats['lr'], epoch)
            writer.add_scalar('Test/loss', test_stats['loss'], epoch)
            writer.add_scalar('Test/Top1', test_stats['acc1'], epoch)
            writer.add_scalar('Test/Top5', test_stats['acc5'], epoch)

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir and dist_utils.is_main_process():
                model_checkpoint = os.path.join(output_dir, f"checkpoint.pth")
                torch.save(model_without_ddp.state_dict(), model_checkpoint)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                with open(os.path.join(output_dir, 'result.txt'), 'w') as f:
                    f.write(f'Final Accuracy: {max_accuracy}\n'
                            f'Model config: {model_config}')
                logger.info(f'Saving model in [PATH]: {output_dir}')
        logger.info(f'Max accuracy: {max_accuracy:.4f}%')

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
    logger.info(f'Training time {total_time_str} on sub-dataset{args.start_division}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
