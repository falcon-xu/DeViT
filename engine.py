# -*- coding: utf-8 -*-
# @Time    : 2023/3/16 9:50
# @Author  : Falcon
# @FileName: engine.py
import torch
import math
import sys
from typing import Iterable, Optional

from utils import dist_utils
from utils.losses import feature_relation_loss
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.utils.clip_grad import dispatch_clip_grad


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

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_1epoch_qkv(model: torch.nn.Module, teacher_model: torch.nn.Module, criterion, data_loader: Iterable,
                     optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, log, args,
                     max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None):
    model.train(mode=True)
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('q_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # q loss
    metric_logger.add_meter('k_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # k loss
    metric_logger.add_meter('v_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # v loss
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.distillation_inter:
                outputs = model(samples, output_qkv=True)
                logits = outputs['output']
                qkvs = outputs['qkv']
                with torch.no_grad():
                    teacher_outputs = teacher_model(samples, output_qkv=True)
                    teacher_logits = teacher_outputs['output']
                    teacher_qkvs = teacher_outputs['qkv']

                # cls loss
                cls_loss = criterion(outputs=logits, teacher_outputs=teacher_logits, labels=targets)

                # qkv loss
                q_loss = 0.
                k_loss = 0.
                v_loss = 0.

                teacher_layer_num = len(teacher_qkvs)
                student_layer_num = len(qkvs)
                assert teacher_layer_num % student_layer_num == 0, 'The number of student layer can not be divisible by the number of teacher layer'
                layers_per_block = int(teacher_layer_num / student_layer_num)

                new_student_qkvs = [qkvs[student_layer_num // 2 - 1]]
                new_teacher_qkvs = [teacher_qkvs[teacher_layer_num // 2 - 1]]

                for student_qkv, teacher_qkv in zip(new_student_qkvs, new_teacher_qkvs):
                    tmp_qkv_loss = [feature_relation_loss(tea_vector, stu_vector) for stu_vector, tea_vector in
                                    zip(student_qkv, teacher_qkv)]
                    tmp_q_loss, tmp_k_loss, tmp_v_loss = tmp_qkv_loss
                    q_loss += tmp_q_loss
                    k_loss += tmp_k_loss
                    v_loss += tmp_v_loss

                q_loss /= student_layer_num
                k_loss /= student_layer_num
                v_loss /= student_layer_num
                loss = cls_loss + float(args.gama[0]) * q_loss + float(args.gama[1]) * k_loss + float(
                    args.gama[2]) * v_loss

                metric_logger.update(cls_loss=cls_loss.item())
                metric_logger.update(q_loss=q_loss.item())
                metric_logger.update(k_loss=k_loss.item())
                metric_logger.update(v_loss=v_loss.item())

            else:
                outputs = model(samples)
                loss = criterion(inputs=samples, outputs=outputs, labels=targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_1epoch_ens_disjoint(model: torch.nn.Module, ens_model: torch.nn.Module, criterion: torch.nn.Module,
                              data_loader: Iterable, optimizer: torch.optim.Optimizer,
                              ens_optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                              scaler, args, log, model_ema: Optional[ModelEma] = None,
                              ens_model_ema: Optional[ModelEma] = None,
                              mixup_fn: Optional[Mixup] = None, max_norm: float = 0):
    model.train(mode=True)
    ens_model.train(mode=True)
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('backbone_lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ens_lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if args.distillation_type != 'none':
        metric_logger.add_meter('cls_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('token_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.distillation_type == 'none':
                features = model(samples)
                logits = ens_model(features)
                loss = criterion(samples, logits, targets)
            else:
                features = model(samples)
                outputs = ens_model(features, True)
                inter_loss, cls_loss = criterion(inputs=samples, stu_outputs=outputs, labels=targets)
                loss = inter_loss + cls_loss
                metric_logger.update(token_loss=inter_loss.item())
                metric_logger.update(cls_loss=cls_loss.item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            log.info(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        ens_optimizer.zero_grad()

        scaler.scale(loss).backward(create_graph=False)
        if max_norm is not None:
            scaler.unscale_(optimizer)
            scaler.unscale_(ens_optimizer)
            dispatch_clip_grad(model.parameters(), max_norm, mode='norm')
            dispatch_clip_grad(ens_model.parameters(), max_norm, mode='norm')
        scaler.step(optimizer)
        scaler.step(ens_optimizer)
        scaler.update()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
            ens_model_ema.update(ens_model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(backbone_lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(ens_lr=ens_optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_ens_disjoint(data_loader, model, ens_model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    ens_model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            output = ens_model(output)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
