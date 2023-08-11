import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

num_blocks = 12

# nn.Linear indices
attn_qkv = [4 * i + 1 for i in range(num_blocks)]
attn_proj = [4 * i + 2 for i in range(num_blocks)]
mlp_fc1 = [4 * i + 3 for i in range(num_blocks)]
mlp_fc2 = [4 * i + 4 for i in range(num_blocks)]


def mlp_neuron_rank(model, train_loader, mode='cuda'):
    relevance = HSICLoss(y_kernel='linear', mean_sub=True).cuda()
    redundancy = HSICLoss(y_kernel='rbf', mean_sub=False).cuda()
    score = {}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 1:
                break
            data, target = Variable(data), Variable(target)
            if mode == 'cuda':
                data, target = data.cuda(), target.cuda()
            output = model(data)
            idx = 0
            for m in model.modules():
                if 'Mlp' in str(m) and 'Attention' not in str(m):
                    X_ = m.neuron_output  # batch x seq x embed
                    hsic = []
                    for H1 in range(X_.shape[-1]):
                        hsic.append(relevance(X_[:, :, H1], F.softmax(output, dim=-1)).item())
                    hsic = np.array(hsic)
                    hsic = (hsic - np.min(hsic)) / (np.max(hsic) - np.min(hsic))
                    act = np.sum(X_.abs().detach().cpu().numpy(), axis=(0, 1))
                    act = (act - np.min(act)) / (np.max(act) - np.min(act))
                    temp = (0.1 * hsic + 0.9 * act).tolist()
                    if batch_idx == 0:
                        score[str(idx)] = np.array(temp)
                    else:
                        score[str(idx)] += np.array(temp)
                    idx += 1
                    continue
    rank = [np.argsort(score[str(idx)]) for idx in range(len(score))]
    return rank


def mlp_neuron_mask(model, ratio, rank):
    idx = 0
    neuron_mask = []
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            num_keep = int(m.hidden_features * (1 - ratio[idx]))
            arg_max_rev = rank[idx][::-1][:num_keep]
            mask = torch.zeros(m.hidden_features)
            mask[arg_max_rev.tolist()] = 1
            neuron_mask.append(mask)
            idx += 1
            continue
    return neuron_mask


def mlp_neuron_prune(model, neuron_mask):
    idx = 0
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            m.gate = neuron_mask[idx]
            idx += 1
            continue


def mlp_neuron_restore(model, ):
    idx = 0
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            temp = m.gate.detach().clone()
            m.gate = torch.ones(temp.shape[0])
            idx += 1
            continue


def check_neuron_sparsity(model):
    ratio = []
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            ratio.append(torch.sum(m.gate == 0).item() / m.gate.shape[0])
            continue
    return ratio


def attn_head_rank(model, train_loader, mode='cuda'):
    relevance = HSICLoss(y_kernel='linear', mean_sub=True).cuda()
    redundancy = HSICLoss(y_kernel='rbf', mean_sub=False).cuda()
    score = {}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 1:
                break
            data, target = Variable(data), Variable(target)
            if mode == 'cuda':
                data, target = data.cuda(), target.cuda()
            output = model(data)
            idx = 0
            for m in model.modules():
                if 'Attention' in str(m) and 'Mlp' not in str(m):
                    X_ = m.head_output  # batch x seq x head x embed_chunk
                    temp = []
                    for H1 in range(X_.shape[2]):
                        # max relevance
                        relevance_count = relevance(torch.mean(X_[:, :, H1, :], dim=-1),
                                                    F.softmax(output, dim=-1)).item()
                        # min redundancy
                        redundancy_count = 0
                        for H2 in range(X_.shape[2]):
                            if H2 != H1:
                                redundancy_count += redundancy(torch.mean(X_[:, :, H1, :], dim=-1),
                                                               torch.mean(X_[:, :, H2, :], dim=-1)).item()
                        redundancy_count /= (X_.shape[2] - 1)
                        temp.append(relevance_count - 0.1 * redundancy_count)
                    if batch_idx == 0:
                        score[str(idx)] = np.array(temp)
                    else:
                        score[str(idx)] += np.array(temp)
                    idx += 1
                    continue
    rank = [(np.argsort(score[str(idx)])) for idx in range(len(score))]
    return rank


def attn_head_mask(model, ratio, rank):
    idx = 0
    head_mask = []
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            num_keep = int(m.num_heads * (1 - ratio[idx]))
            arg_max_rev = rank[idx][::-1][:num_keep]
            mask = torch.zeros(m.num_heads)
            mask[arg_max_rev.tolist()] = 1
            head_mask.append(mask)
            idx += 1
            continue
    return head_mask


def attn_head_prune(model, head_mask):
    idx = 0
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            m.gate = head_mask[idx]
            idx += 1
            continue


def attn_head_restore(model, ):
    idx = 0
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            temp = m.gate.detach().clone()
            m.gate = torch.ones(temp.shape[0])
            idx += 1
            continue


def check_head_sparsity(model):
    ratio = []
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            ratio.append(torch.sum(m.gate == 0).item() / m.gate.shape[0])
            continue
    return ratio


def actual_prune(model, data_loader_train, neuron_sparsity, head_sparsity, log, finetune_path=''):
    # get rank of mlp neuron and head
    mlp_rank = mlp_neuron_rank(model, data_loader_train)
    att_rank = attn_head_rank(model, data_loader_train)
    # get mask of mlp neuron and head
    neuron_mask = mlp_neuron_mask(model, neuron_sparsity, mlp_rank)
    head_mask = attn_head_mask(model, head_sparsity, att_rank)

    # load finetune ckpt
    if finetune_path != '':
        model.load_state_dict(torch.load(finetune_path, map_location='cpu'))
        log.info(f'Load pretrained checkpoint from [PATH]: {finetune_path}')

    # neuron actual prune
    idx = 0
    for m in model.modules():
        if 'Mlp' in str(m) and 'Attention' not in str(m):
            m.gate = neuron_mask[idx]
            m.prune(neuron_mask[idx])
            idx += 1

    # head actual prune
    idx = 0
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            m.gate = head_mask[idx]
            m.prune(head_mask[idx])
            idx += 1


def set_token_selection_layer(model, token_sparsity):
    idx = 1
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            if idx in [4, 7, 10]:
                m.token_prune_ratio = token_sparsity
            idx += 1
            continue


def reset_token_selection_layer(model, ):
    idx = 1
    for m in model.modules():
        if 'Attention' in str(m) and 'Mlp' not in str(m):
            if idx in [4, 7, 10]:
                m.token_prune_ratio = 0
            idx += 1
            continue
    for m in model.modules():
        if 'Block' in str(m) and 'ModuleList' not in str(m):
            m.ema_cls_attn = None


def center(X):
    mean_col = torch.mean(X, dim=0, keepdim=True)
    mean_row = torch.mean(X, dim=1, keepdim=True)
    mean_all = torch.mean(X)
    return X - mean_col - mean_row + mean_all


class GaussianKernel(nn.Module):
    def __init__(self, sigma):
        super(GaussianKernel, self).__init__()
        assert sigma > 0
        self.sigma = sigma

    def forward(self, x):
        X_inner = torch.matmul(x, x.t())
        X_norm = torch.diag(X_inner, diagonal=0)
        X_dist_sq = X_norm + torch.reshape(X_norm, [-1, 1]) - 2 * X_inner
        return torch.exp(- X_dist_sq / (2 * self.sigma ** 2))


class LinearKernel(nn.Module):
    def __init__(self, ):
        super(LinearKernel, self).__init__()

    def forward(self, x):
        return torch.matmul(x, x.t())


class HSICLoss(nn.Module):
    def __init__(self, y_kernel='linear', mean_sub=False):
        super(HSICLoss, self).__init__()

        self.kernelX_1 = GaussianKernel(1)
        self.kernelX_2 = GaussianKernel(2)
        self.kernelX_4 = GaussianKernel(4)
        self.kernelX_8 = GaussianKernel(8)
        self.kernelX_16 = GaussianKernel(16)

        self.y_kernel = y_kernel
        if self.y_kernel == 'linear':
            self.kernelY = LinearKernel()
        elif self.y_kernel == 'rbf':
            self.kernelY = None

        self.mean_sub = mean_sub

    def forward(self, x, y):
        '''
        x: feature
        y: softmax prediction
        '''
        if self.mean_sub is True:
            x = x - torch.mean(x, dim=0) / (torch.std(x, dim=0) + 1e-12)
            y = y - torch.mean(y, dim=0)

        G_X = center(
            (self.kernelX_1(x) + self.kernelX_2(x) + self.kernelX_4(x) + self.kernelX_8(x) + self.kernelX_16(x)) / 5)

        if self.y_kernel == 'linear':
            G_Y = center(self.kernelY(y))
        elif self.y_kernel == 'rbf':
            G_Y = center((self.kernelX_1(y) + self.kernelX_2(y) + self.kernelX_4(y) + self.kernelX_8(
                y) + self.kernelX_16(y)) / 5)

        return torch.trace(torch.matmul(G_X, G_Y))
