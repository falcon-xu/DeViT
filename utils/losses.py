# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def _compute_losses(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        loss = (- targets_prob * student_likelihood).mean()
        return loss

    def forward(self, predicts, targets):
        return self._compute_losses(predicts, targets)


class DistillLoss(nn.Module):
    '''
    KD loss without teacher model
    '''

    def __init__(self, base_criterion: torch.nn.Module, distillation_type: str, alpha: float, tau: float):
        super(DistillLoss, self).__init__()
        self.base_criterion = base_criterion
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def _compute_cls_distill_loss(self, outputs_kd, teacher_outputs):
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                # We provide the teacher's targets in log probability because we use log_target=True
                # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        return distillation_loss

    def forward(self, outputs, teacher_outputs, labels):
        '''
        Args:
            outputs (): student outputs
            teacher_outputs (): teacher outputs
            labels ():
        Returns:
        '''
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        else:
            outputs_kd = outputs

        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        distillation_loss = self._compute_cls_distill_loss(outputs_kd, teacher_outputs)
        cls_loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return cls_loss


class EnsLoss(nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module, model,
                 distillation_type: str, alpha: float, tau: float, loss_type='mse'):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.model = model
        if loss_type == 'mse':
            self.token_criterion = nn.MSELoss()
        elif loss_type == 'kldiv':
            self.token_criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def _compute_cls_distill_loss(self, outputs_kd, teacher_outputs):
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                # We provide the teacher's targets in log probability because we use log_target=True
                # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        return distillation_loss

    def forward(self, inputs, stu_outputs, labels):
        if self.distillation_type == 'none':
            base_cls_loss = self.base_criterion(stu_outputs, labels)
            return base_cls_loss
        else:
            with torch.no_grad():
                tea_outputs = self.teacher_model(inputs, distill_token=True)

            if 'vit' in self.model:
                stu_token, stu_logits = stu_outputs
                tea_logits = tea_outputs['output']
                tea_token = tea_outputs['last_tokens']
                token_loss = self.token_criterion(stu_token, tea_token)
                cls_loss = (1 - self.alpha) * self.base_criterion(stu_logits, labels) + \
                           self.alpha * self._compute_cls_distill_loss(stu_logits, tea_logits)
                return token_loss, cls_loss

            elif 'deit' in self.model:
                tokens, stu_logits = stu_outputs
                cls_token, dist_token = tokens
                tea_logits = tea_outputs['output']
                tea_token, tea_token_dist = tea_outputs['last_tokens']

                cls_loss = (1 - self.alpha) * self.base_criterion(stu_logits, labels) + \
                           self.alpha * self._compute_cls_distill_loss(stu_logits, tea_logits)
                token_loss = self.token_criterion(cls_token, tea_token) + \
                             self.token_criterion(dist_token, tea_token_dist)

                return token_loss, cls_loss


def cal_qkv_loss(stu_qkv_list, tea_qkv_list):

    layer_num = len(stu_qkv_list)
    loss = 0.

    for stu_qkv, tea_qkv in zip(stu_qkv_list, tea_qkv_list):
        B, Hs, N, Cs = stu_qkv[0].shape
        _, Ht, _, Ct = tea_qkv[0].shape

        for i in range(3):
            Ms = stu_qkv[i].contiguous().view(B, N, Hs * Cs)
            Ms1 = Ms / Cs**0.5
            Ms2 = Ms.transpose(1, 2)
            Ms12 = (Ms1 @ Ms2)

            Mt = tea_qkv[i].contiguous().view(B, N, Ht * Ct)
            Mt1 = Mt / Ct**0.5
            Mt2 = Mt.transpose(1, 2)
            Mt12 = (Mt1 @ Mt2)

            loss += soft_cross_entropy(Ms12, Mt12)
    return loss / (3. * layer_num)


def cal_qkv_loss2(stu_qkv_list, tea_qkv_list):

    layer_num = len(stu_qkv_list)
    loss = 0.

    for stu_qkv, tea_qkv in zip(stu_qkv_list, tea_qkv_list):
        B, Hs, N, Cs = stu_qkv[0].shape
        _, Ht, _, Ct = tea_qkv[0].shape

        for i in range(3):
            for j in range(3):

                Ms1 = stu_qkv[i].contiguous().view(B, N, Hs * Cs) / Cs**0.5
                Ms2 = stu_qkv[j].contiguous().view(B, N, Hs * Cs).transpose(1, 2)
                Ms12 = (Ms1 @ Ms2)

                Mt1 = tea_qkv[i].contiguous().view(B, N, Ht * Ct) / Ct ** 0.5
                Mt2 = tea_qkv[j].contiguous().view(B, N, Ht * Ct).transpose(1, 2)
                Mt12 = (Mt1 @ Mt2)

                loss += soft_cross_entropy(Ms12, Mt12)
    return loss / (9. * layer_num)


def cal_hid_relation_loss(stu_hid_list, tea_hid_list):
    layer_num = len(stu_hid_list)
    loss = 0.
    for stu_hid, tea_hid in zip(stu_hid_list, tea_hid_list):
        stu_hid = torch.nn.functional.normalize(stu_hid, dim=-1)
        tea_hid = torch.nn.functional.normalize(tea_hid, dim=-1)
        stu_rela = stu_hid @ stu_hid.transpose(-1, -2)
        tea_rela = tea_hid @ tea_hid.transpose(-1, -2)
        loss += torch.mean((stu_rela - tea_rela)**2)
    return loss / layer_num