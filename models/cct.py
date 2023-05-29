'''
Refer from https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/cct.py
'''


import torch
from torch.hub import load_state_dict_from_url
import torch.nn as nn
from .utils.transformers import TransformerClassifier, CCTTransformer
from .utils.tokenizer import Tokenizer
from .utils.helpers import pe_check

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

model_urls = {
    'cct_7_3x1_32':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth',
    'cct_7_3x1_32_sine':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_5000epochs.pth',
    'cct_7_3x1_32_c100':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs.pth',
    'cct_7_3x1_32_sine_c100':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_5000epochs.pth',
    'cct_7_7x2_224_sine':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth',
    'cct_14_7x2_224':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
    'cct_14_7x2_384':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
    'cct_14_7x2_384_fl':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth',
}


class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 resize_dim=None,
                 backbone=False,
                 *args, **kwargs):
        '''

        Args:
            img_size ():
            embedding_dim ():
            n_input_channels ():
            n_conv_layers ():
            kernel_size ():
            stride ():
            padding ():
            pooling_kernel_size ():
            pooling_stride ():
            pooling_padding ():
            dropout ():
            attention_dropout ():
            stochastic_depth ():
            num_layers ():
            num_heads ():
            mlp_ratio ():
            num_classes ():
            positional_embedding ():
            resize_dim (int): resize intermediate feature dimension to match the output feature of teacher model
            *args ():
            **kwargs ():
        '''
        super(CCT, self).__init__()
        self.backbone = backbone

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        if backbone:
            self.encoders = CCTTransformer(
                sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                               height=img_size,
                                                               width=img_size),
                embedding_dim=embedding_dim,
                seq_pool=True,
                dropout=float(dropout),
                attention_dropout=attention_dropout,
                stochastic_depth=stochastic_depth,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_classes=num_classes,
                positional_embedding=positional_embedding)
        else:
            self.classifier = TransformerClassifier(
                sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                               height=img_size,
                                                               width=img_size),
                embedding_dim=embedding_dim,
                seq_pool=True,
                dropout=float(dropout),
                attention_dropout=attention_dropout,
                stochastic_depth=stochastic_depth,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_classes=num_classes,
                positional_embedding=positional_embedding
            )
        self.resize_dim = resize_dim
        if self.resize_dim is not None:
            # TODO: add conv to resize height. now only the feature with equal height can be distilled
            self.resize = nn.Linear(embedding_dim, resize_dim)

    def forward(self, x, output_attention=False, output_hidden_states=False, output_pool=False, distill=False):
        '''

        Args:
            x ():
            output_attention ():
            output_hidden_states ():
            distill (bool): whether to distill

        Returns:

        '''

        x = self.tokenizer(x)
        # initial code
        # return self.classifier(x)

        if self.backbone:
            outputs = self.encoders(x)
            return outputs
        else:
            # modified edition
            outputs = self.classifier(x, output_attention=output_attention, output_hidden_states=output_attention,
                                      output_pool=output_pool)

            if distill:
                logits = outputs[0]
                atts_output = outputs[1]
                hids_output = outputs[2]
                if self.resize_dim is not None:
                    atts_output = tuple(self.resize(att_output) for att_output in atts_output)
                    hids_output = tuple(self.resize(hid_output) for hid_output in hids_output)
                outputs = (logits, atts_output, hids_output)

            if (not output_attention) and (not output_hidden_states) and (not output_pool):
                return outputs[0]
            else:
                return outputs

    def reset_classifier(self, num_classes):
        self.classifier.reset_classifier(num_classes=num_classes)


def _cct(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None, resize_dim=None,backbone=False,
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                resize_dim=resize_dim,
                backbone=backbone,
                *args, **kwargs)

    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            state_dict = pe_check(model, state_dict)
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f'Variant {arch} does not yet have pretrained weights.')
    return model


def construct_cct(config, resize_dim=None, arch=None, pretrained=False, progress=True, *args, **kwargs):
    config['stride'] = config['stride'] if config['stride'] is not None else max(1, (config['kernel_size'] // 2) - 1)
    config['padding'] = config['padding'] if config['padding'] is not None else max(1, (config['kernel_size'] // 2))
    config['resize_dim'] = resize_dim
    model = CCT(**config)

    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            state_dict = pe_check(model, state_dict)
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f'Variant {arch} does not yet have pretrained weights.')
    return config, model


def cct_2(arch=None, pretrained=False, progress=True, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(arch=None, pretrained=False, progress=True, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(arch=None, pretrained=False, progress=True, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(arch=None, pretrained=False, progress=True, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(arch=None, pretrained=False, progress=True, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def get_cct2(pretrained_path=None, num_classes=1000, progress=False, kernel_size=3, n_conv_layers=2,
             img_size=32, positional_embedding='learnable', *args, **kwargs):
    model = cct_2(pretrained=False, progress=progress,
                  kernel_size=kernel_size, n_conv_layers=n_conv_layers,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))

    return model


def get_cct6(pretrained_path=None, num_classes=1000, progress=False, kernel_size=3, n_conv_layers=2,
             img_size=32, positional_embedding='learnable', *args, **kwargs):
    model = cct_6(pretrained=False, progress=progress,
                  kernel_size=kernel_size, n_conv_layers=n_conv_layers,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))

    return model


def get_cct7(pretrained_path=None, num_classes=1000, progress=False, kernel_size=3, n_conv_layers=2,
             img_size=32, positional_embedding='learnable', backbone=False, *args, **kwargs):
    model = cct_7(pretrained=False, progress=progress,
                  kernel_size=kernel_size, n_conv_layers=n_conv_layers,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes, backbone=backbone,
                  *args, **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))

    return model


def get_cct14(pretrained_path=None, num_classes=1000, progress=False, kernel_size=3, n_conv_layers=2,
              img_size=32, positional_embedding='learnable', *args, **kwargs):
    model = cct_14(pretrained=False, progress=progress,
                   kernel_size=kernel_size, n_conv_layers=n_conv_layers,
                   img_size=img_size, positional_embedding=positional_embedding,
                   num_classes=num_classes,
                   *args, **kwargs)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))

    return model


@register_model
def cct_2_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_2('cct_2_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_2_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_2('cct_2_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_4('cct_4_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_4_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_4('cct_4_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_6('cct_6_3x1_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x1_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_6('cct_6_3x1_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_6('cct_6_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_6_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_6('cct_6_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x1_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_7('cct_7_3x1_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_c100(pretrained=False, progress=False,
                      img_size=32, positional_embedding='learnable', num_classes=100,
                      *args, **kwargs):
    return cct_7('cct_7_3x1_32_c100', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x1_32_sine_c100(pretrained=False, progress=False,
                           img_size=32, positional_embedding='sine', num_classes=100,
                           *args, **kwargs):
    return cct_7('cct_7_3x1_32_sine_c100', pretrained, progress,
                 kernel_size=3, n_conv_layers=1,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x2_32(pretrained=False, progress=False,
                 img_size=32, positional_embedding='learnable', num_classes=10,
                 *args, **kwargs):
    return cct_7('cct_7_3x2_32', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_3x2_32_sine(pretrained=False, progress=False,
                      img_size=32, positional_embedding='sine', num_classes=10,
                      *args, **kwargs):
    return cct_7('cct_7_3x2_32_sine', pretrained, progress,
                 kernel_size=3, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224(pretrained=False, progress=False,
                  img_size=224, positional_embedding='learnable', num_classes=102,
                  *args, **kwargs):
    return cct_7('cct_7_7x2_224', pretrained, progress,
                 kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_7_7x2_224_sine(pretrained=False, progress=False,
                       img_size=224, positional_embedding='sine', num_classes=102,
                       *args, **kwargs):
    return cct_7('cct_7_7x2_224_sine', pretrained, progress,
                 kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)


@register_model
def cct_14_7x2_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return cct_14('cct_14_7x2_224', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_14_7x2_384(pretrained=False, progress=False,
                   img_size=384, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return cct_14('cct_14_7x2_384', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def cct_14_7x2_384_fl(pretrained=False, progress=False,
                      img_size=384, positional_embedding='learnable', num_classes=102,
                      *args, **kwargs):
    return cct_14('cct_14_7x2_384_fl', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)
