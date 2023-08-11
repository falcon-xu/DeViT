

def cal_prune_paras(neuron_sparsity: list, head_sparsity: list,
                    emb=768, seq_length=197, mlp_ratio=4, head=12, layer=12, num_class=1000):
    assert len(head_sparsity) == layer, 'The number of layer is not equal to the number of head sparsity.'

    paras = 0
    channel = 3
    patch_size = 16
    head_dim = emb / head
    # Embedding: 1) Patch Embedding(Conv); 2) Position Embedding; 3) CLS token
    paras += emb * channel * patch_size ** 2 + emb + seq_length * emb + emb

    ln = 2 * emb

    for n_s, h_s in zip(neuron_sparsity, head_sparsity):
        prune_head = int((1 - h_s) * head)

        # Encoder: 1) LN; 2) MHSA; 3) LN; 4) MLP
        mhsa = prune_head * 3 * emb * head_dim + prune_head * head_dim * emb + emb
        mlp = 2 * emb * int(mlp_ratio * (1 - n_s) * emb) + emb + int(mlp_ratio * (1 - n_s) * emb)

        paras += ln + mhsa + ln + mlp

    # Classification head
    cls = emb * num_class + num_class

    paras += ln + cls

    return paras / 1e6


def cal_prune_flops(neuron_sparsity: list, head_sparsity: list,
                    emb=768, seq_length=197, mlp_ratio=4, head=12, layer=12, num_class=1000):
    assert len(head_sparsity) == layer, 'The number of layer is not equal to the number of head sparsity.'

    # Neglect softmax and norm
    flops = 0
    channel = 3
    patch_size = 16
    img_size = 224
    head_dim = emb / head
    # Embedding
    flops += 2 * channel * emb * img_size ** 2

    for n_s, h_s in zip(neuron_sparsity, head_sparsity):
        # MHSA: 1) QKV projection; 2) Q*K; 3) (Q*K)*V; 4) linear projection for the concatenated

        sa = 3 * 2 * seq_length * emb * head_dim + \
             2 * head_dim * seq_length ** 2 + \
             2 * head_dim * seq_length ** 2

        prune_head = int((1 - h_s) * head)

        mhsa = sa * prune_head + seq_length * 2 * head_dim * prune_head * emb

        # MLP: 1) Linear 1 ; 2) Linear 2
        mlp = seq_length * int(mlp_ratio * (1 - n_s) * emb) * 2 * emb + \
              seq_length * emb * 2 * int(mlp_ratio * (1 - n_s) * emb)

        flops += mhsa + mlp

    # Classification head
    flops += 2 * emb * num_class

    return flops / 1e9


def cal_prune_macs(neuron_sparsity: list, head_sparsity: list,
                    emb=768, seq_length=197, mlp_ratio=4, head=12, layer=12, num_class=1000):
    return cal_prune_flops(neuron_sparsity,head_sparsity,emb,seq_length,mlp_ratio,head,layer,num_class) / 2


