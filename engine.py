# -*- coding: utf-8 -*-
# @Time    : 2023/3/16 9:50
# @Author  : Falcon
# @FileName: engine.py
import torch
import math
import sys
from typing import Iterable, Optional

from utils import dist_utils
from utils.losses import feature_relation_loss, manifold_loss, cal_qkv_loss, cal_hid_relation_loss, cal_qkv_loss2
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


def train_1epoch_qkv_new(model: torch.nn.Module, teacher_model: torch.nn.Module, criterion, data_loader: Iterable,
                         optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, log, args,
                         max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None):
    model.train(mode=True)
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('qkv_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # qkv loss
    metric_logger.add_meter('hid_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.distillation_inter:
                outputs = model(samples, output_qkv=True, output_encoders=True)
                logits = outputs['output']
                qkvs = outputs['qkv']
                hids = outputs['encoder']

                with torch.no_grad():
                    teacher_outputs = teacher_model(samples, output_qkv=True, output_encoders=True)
                teacher_logits = teacher_outputs['output']
                teacher_qkvs = teacher_outputs['qkv']
                teacher_hids = teacher_outputs['encoder']

                # cls loss
                cls_loss = criterion(outputs=logits, teacher_outputs=teacher_logits, labels=targets)

                # qkv loss
                qkv_loss = cal_qkv_loss(qkvs, teacher_qkvs)

                # hid loss
                hid_loss = cal_hid_relation_loss(hids, teacher_hids)

                loss = cls_loss + float(args.gama[0]) * qkv_loss + float(args.gama[1]) * hid_loss

                metric_logger.update(cls_loss=cls_loss.item())
                metric_logger.update(qkv_loss=qkv_loss.item())
                metric_logger.update(hid_loss=hid_loss.item())

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


def train_1epoch_qkv2(model: torch.nn.Module, teacher_model: torch.nn.Module, criterion, data_loader: Iterable,
                      optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, log, args,
                      max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None):
    model.train(mode=True)
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('qkv_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # qkv loss
    metric_logger.add_meter('hid_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.distillation_inter:
                outputs = model(samples, output_qkv=True, output_encoders=True)
                logits = outputs['output']
                qkvs = outputs['qkv']
                hids = outputs['encoder']

                with torch.no_grad():
                    teacher_outputs = teacher_model(samples, output_qkv=True, output_encoders=True)
                teacher_logits = teacher_outputs['output']
                teacher_qkvs = teacher_outputs['qkv']
                teacher_hids = teacher_outputs['encoder']

                # cls loss
                cls_loss = criterion(outputs=logits, teacher_outputs=teacher_logits, labels=targets)

                # qkv loss
                qkv_loss = cal_qkv_loss2(qkvs, teacher_qkvs)

                # hid loss
                hid_loss = cal_hid_relation_loss(hids, teacher_hids)

                loss = cls_loss + float(args.gama[0]) * qkv_loss + float(args.gama[1]) * hid_loss

                metric_logger.update(cls_loss=cls_loss.item())
                metric_logger.update(qkv_loss=qkv_loss.item())
                metric_logger.update(hid_loss=hid_loss.item())

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


def train_1epoch_mf(args, model: torch.nn.Module, teacher_model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    loss_scaler, log, max_norm: float = 0, model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None):
    model.train(mode=True)
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('cls_loss', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mf_patch_loss',
                            dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # manifold patch loss
    metric_logger.add_meter('mf_sample_loss',
                            dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # manifold sample loss
    metric_logger.add_meter('mf_rand_loss',
                            dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # manifold rand loss
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.distillation_inter:
                outputs = model(samples, output_att=True, output_emb=True, output_encoders=True)
                with torch.no_grad():
                    teacher_outputs = teacher_model(samples, output_att=True, output_emb=True, output_encoders=True)

                logits = outputs['output']
                teacher_logits = teacher_outputs['output']

                # cls loss, cls_token loss
                cls_loss = criterion(outputs=logits, teacher_outputs=teacher_logits, labels=targets)

                loss_patch = 0.
                loss_sample = 0.
                loss_rand = 0.

                student_atts, student_hids = list(outputs.values())[2:4]
                teacher_atts, teacher_hids = list(teacher_outputs.values())[2:4]
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                assert teacher_layer_num % student_layer_num == 0, 'The number of student layer can not be divisible by the number of teacher layer'
                layers_per_block = int(teacher_layer_num / student_layer_num)

                new_teacher_hids = [teacher_hids[i * layers_per_block] for i in range(student_layer_num + 1)]
                for student_hid, teacher_hid in zip(student_hids, new_teacher_hids):
                    tmp_loss = manifold_loss(teacher_feature=teacher_hid, student_feature=student_hid, mode='kldiv')
                    loss_patch += tmp_loss[0]
                    loss_sample += tmp_loss[1]
                    loss_rand += tmp_loss[2]

                loss = cls_loss + int(args.gama[0]) * loss_patch + int(args.gama[1]) * loss_sample + int(
                    args.gama[2]) * loss_rand
                metric_logger.update(cls_loss=cls_loss.item())
                metric_logger.update(mf_patch_loss=loss_patch.item())
                metric_logger.update(mf_sample_loss=loss_sample.item())
                metric_logger.update(mf_rand_loss=loss_rand.item())

            else:
                outputs = model(samples)
                loss = criterion(inputs=samples, outputs=outputs, labels=targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            log.info(f"Loss is {loss_value}, stopping training")
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


def train_1epoch_ensemble(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
                          optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, args, log,
                          model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, max_norm: float = 0):
    model.train(mode=True)
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
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
                logits = model(samples)
                loss = criterion(samples, logits, targets)
            else:
                outputs = model(samples, True)
                inter_loss, cls_loss = criterion(inputs=samples, stu_outputs=outputs, labels=targets)
                loss = inter_loss + cls_loss
                metric_logger.update(token_loss=inter_loss.item())
                metric_logger.update(cls_loss=cls_loss.item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            log.info(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

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


def train_1epoch_ens_freeze(model: torch.nn.Module, ens_model: torch.nn.Module, criterion: torch.nn.Module,
                            data_loader: Iterable, ens_optimizer: torch.optim.Optimizer, device: torch.device,
                            epoch: int, loss_scaler, args, log, ens_model_ema: Optional[ModelEma] = None,
                            mixup_fn: Optional[Mixup] = None, max_norm: float = 0):
    # model.train(mode=True)
    ens_model.train(mode=True)
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
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
            with torch.no_grad():
                features = model(samples)

            if args.distillation_type == 'none':
                # features = model(samples)
                logits = ens_model(features)
                loss = criterion(samples, logits, targets)
            else:
                # features = model(samples)
                outputs = ens_model(features, True)
                inter_loss, cls_loss = criterion(inputs=samples, stu_outputs=outputs, labels=targets)
                loss = inter_loss + cls_loss
                metric_logger.update(token_loss=inter_loss.item())
                metric_logger.update(cls_loss=cls_loss.item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            log.info(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        ens_optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(ens_optimizer, 'is_second_order') and ens_optimizer.is_second_order
        loss_scaler(loss, ens_optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph=is_second_order)

        torch.cuda.synchronize()
        if ens_model_ema is not None:
            ens_model_ema.update(ens_model)

        metric_logger.update(loss=loss_value)
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
