import time
from timm.utils import accuracy
from .compute_metric import cal_shrink_paras, cal_shrink_macs
from .imp_rank import *

num_blocks = 12

# nn.Linear indices
attn_qkv = [4 * i + 1 for i in range(num_blocks)]
attn_proj = [4 * i + 2 for i in range(num_blocks)]
mlp_fc1 = [4 * i + 3 for i in range(num_blocks)]
mlp_fc2 = [4 * i + 4 for i in range(num_blocks)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


@torch.no_grad()
def shrink_evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()

    for _, (images, target) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        top1.update(acc1.item(), batch_size)

    return top1.avg


def screen(macs_target, population, lb, ub, layer, log):
    start_time = time.time()
    res = []
    n_params = layer * 2
    while len(res) < population:
        ratio = (np.random.uniform(lb, ub, size=(1, n_params))[0]).tolist()

        macs = cal_shrink_macs(
            neuron_sparsity=ratio[:layer], head_sparsity=ratio[layer:],
            emb=384, mlp_ratio=4, seq_length=197, head=6, layer=layer)

        if abs(macs - macs_target) <= 0.02 * macs_target:
            if ratio not in res:
                res.append(ratio)
            log.info(f'#samples: {len(res)}')
    log.info(f'Sampling time: {time.time() - start_time}')
    return res


def random_point(macs_target, population, lb, ub, n_params):
    res = []
    layer = int(n_params / 2)
    while len(res) < population:
        ratio = (np.random.uniform(lb, ub, size=(1, n_params))[0]).tolist()

        macs = cal_shrink_macs(
            neuron_sparsity=ratio[:layer], head_sparsity=ratio[layer:],
            emb=384, mlp_ratio=4, seq_length=197, head=6, layer=layer)

        if abs(macs - macs_target) <= 0.02 * macs_target:
            if ratio not in res:
                res.append(ratio)
    return res[0]


def macs_constraint(x, embed=384, mlp_ratio=4, seq_length=197, head=6, layer=12, num_class=1000):
    neuron_sparsity = x[:layer]
    head_sparsity = x[layer:]

    # Neglect softmax and norm
    flops = 0
    channel = 3
    patch_size = 16
    img_size = 224
    head_dim = embed / head
    # Embedding
    flops += 2 * channel * embed * img_size ** 2

    for n_s, h_s in zip(neuron_sparsity, head_sparsity):
        # MHSA: 1) QKV projection; 2) Q*K; 3) (Q*K)*V; 4) linear projection for the concatenated

        sa = 3 * 2 * seq_length * embed * head_dim + \
             2 * head_dim * seq_length ** 2 + \
             2 * head_dim * seq_length ** 2

        shrink_head = int((1 - h_s) * head)

        mhsa = sa * shrink_head + seq_length * 2 * head_dim * shrink_head * embed

        # MLP: 1) Linear 1 ; 2) Linear 2
        mlp = seq_length * int(mlp_ratio * (1 - n_s) * embed) * 2 * embed + \
              seq_length * embed * 2 * int(mlp_ratio * (1 - n_s) * embed)

        flops += mhsa + mlp

    # Classification head
    flops += 2 * embed * num_class
    macs = flops / 2

    return macs / 1e9


def model_shrink(
        model, data_loader_val, layer, shrink_ratio, neuron_rank, head_rank, device, population, lb, ub, log
):
    x_list = []
    y_list = []

    macs_target = shrink_ratio * 9.19

    # initialize the population
    candidate_set = screen(macs_target=macs_target, population=population, lb=lb, ub=ub, layer=layer,
                           log=log)
    for ratio in candidate_set:
        mlp_neuron_shrink(model.module, mlp_neuron_mask(model.module, ratio[:layer], neuron_rank))
        attn_head_shrink(model.module, attn_head_mask(model.module, ratio[layer:], head_rank))

        macs = cal_shrink_macs(
            neuron_sparsity=ratio[:layer],
            head_sparsity=ratio[layer:],
            emb=384, mlp_ratio=4, seq_length=197, head=6, layer=layer)
        paras = cal_shrink_paras(
            neuron_sparsity=ratio[:layer],
            head_sparsity=ratio[layer:],
            emb=384, mlp_ratio=4, seq_length=197, head=6, layer=layer)

        reward = shrink_evaluate(data_loader_val, model, device)
        log.info('\n-------------------------------------------------\n'
                 f'Neuron sparsity: {ratio[:layer]}\n'
                 f'Head sparsity: {ratio[layer:]}\n'
                 f'Accuracy: {reward}'
                 f'New MACs: {macs} G\n'
                 f'New Parameters: {paras} M')

        mlp_neuron_restore(model.module)
        attn_head_restore(model.module)

        x_list.append(np.array(ratio))
        y_list.append(reward)

    xp = np.array(x_list)
    yp = np.array(y_list)

    return xp, yp
