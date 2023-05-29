# -*- coding: utf-8 -*-
# @Time    : 2023/3/22 21:57
# @Author  : Falcon
# @FileName: ensemble_models.py

import torch
import torch.nn as nn
from timm.models import create_model
# import de_vit
from .de_vit import *


class MultiModels(nn.Module):

    def __init__(self, model='vit_tiny_patch16_224', drop=0, drop_path=0.1, num_class=100,
                 num_classes_list=[25, 25, 25, 25], num_div=4, sub_size=192, teacher_size=None):
        super().__init__()
        self.sub_size = sub_size
        self.teacher_size = teacher_size
        self.model = model
        assert len(num_classes_list) == num_div, 'num of classes is not match num of sub-models'
        self.num_classes = num_class
        self.sum_feature_dim = self.sub_size * len(num_classes_list)

        self.backbones = nn.ModuleList([])
        for i, num_class in enumerate(num_classes_list):
            self.backbones.append(create_model(model_name=self.model,
                                               num_classes=int(num_class),
                                               drop_rate=drop,
                                               drop_path_rate=drop_path,
                                               drop_block_rate=None, ))
            del self.backbones[i].head
            if 'deit' in self.model:
                del self.backbones[i].head_dist

        if self.teacher_size is None:
            self.cls_classifier = nn.Linear(self.sum_feature_dim, self.num_classes)
        else:
            self.cls_mlp = nn.Linear(self.sum_feature_dim, self.teacher_size)
            self.cls_classifier = nn.Linear(self.teacher_size, self.num_classes)
            if 'deit' in self.model:
                self.dist_mlp = nn.Linear(self.sum_feature_dim, self.teacher_size)
                self.dist_classifier = nn.Linear(self.teacher_size, self.num_classes)

    def forward(self, x, distill=False):
        output_features = [model.forward_features(x) for model in self.backbones]
        if 'vit' in self.model:
            output_cls_list = [output_feature['output'] for output_feature in output_features]
            ens_dist_tokens = None
        else:
            output_cls_list = [output_feature['output'][0] for output_feature in output_features]
            output_dist_list = [output_feature['output'][1] for output_feature in output_features]
            ens_dist_tokens = torch.stack(output_dist_list, 1).view(output_dist_list[0].shape[0], -1)

        ens_cls_tokens = torch.stack(output_cls_list, 1).view(output_cls_list[0].shape[0], -1)

        if self.teacher_size is not None:
            cls_token_mlp = self.cls_mlp(ens_cls_tokens)
            logits = self.cls_classifier(cls_token_mlp)
            token_mlp = cls_token_mlp
            if 'deit' in self.model and ens_dist_tokens is not None:
                dist_token_mlp = self.dist_mlp(ens_dist_tokens)
                logits = (logits + self.dist_classifier(dist_token_mlp)) / 2
                token_mlp = (cls_token_mlp, dist_token_mlp)
        elif self.teacher_size is None and 'deit' in self.model:
            logits = (self.cls_classifier(ens_cls_tokens) + self.dist_classifier(ens_dist_tokens)) / 2
        else:
            logits = self.cls_classifier(ens_cls_tokens)

        if distill and self.training and self.teacher_size is not None:
            return token_mlp, logits
        else:
            return logits


class MultiModels1(nn.Module):

    def __init__(self, model='vit_tiny_patch16_224', drop=0, drop_path=0.1, num_class=100,
                 num_classes_list=[25, 25, 25, 25], num_div=4, sub_size=192, teacher_size=None):
        super().__init__()
        self.sub_size = sub_size
        self.teacher_size = teacher_size
        self.model = model
        assert len(num_classes_list) == num_div, 'num of classes is not match num of sub-models'
        self.num_classes = num_class
        self.sum_feature_dim = self.sub_size * len(num_classes_list)

        self.backbones = nn.ModuleList([])
        for i, num_class in enumerate(num_classes_list):
            self.backbones.append(create_model(model_name=self.model,
                                               num_classes=int(num_class),
                                               drop_rate=drop,
                                               drop_path_rate=drop_path,
                                               drop_block_rate=None, ))
            del self.backbones[i].head
            if 'deit' in self.model:
                del self.backbones[i].head_dist

        if self.teacher_size is None:
            self.cls_classifier = nn.Linear(self.sum_feature_dim, self.num_classes)
            if 'deit' in self.model:
                self.dist_classifier = nn.Linear(self.sum_feature_dim, self.num_classes)
        else:
            self.cls_mlp = nn.Linear(self.sum_feature_dim, self.teacher_size)
            self.cls_classifier = nn.Linear(self.teacher_size, self.num_classes)
            if 'deit' in self.model:
                self.dist_mlp = nn.Linear(self.sum_feature_dim, self.teacher_size)
                self.dist_classifier = nn.Linear(self.teacher_size, self.num_classes)

    def forward(self, x, distill=False):
        output_features = [model.forward_features(x) for model in self.backbones]
        if 'vit' in self.model:
            output_cls_list = [output_feature['output'] for output_feature in output_features]
            ens_dist_tokens = None
        else:
            output_cls_list = [output_feature['output'][0] for output_feature in output_features]
            output_dist_list = [output_feature['output'][1] for output_feature in output_features]
            ens_dist_tokens = torch.stack(output_dist_list, 1).view(output_dist_list[0].shape[0], -1)

        ens_cls_tokens = torch.stack(output_cls_list, 1).view(output_cls_list[0].shape[0], -1)

        if 'vit' in self.model:

            if self.teacher_size is not None:
                ens_cls_tokens = self.cls_mlp(ens_cls_tokens)

            ens_token = ens_cls_tokens
            logits = self.cls_classifier(ens_cls_tokens)

        elif 'deit' in self.model:

            if self.teacher_size is not None:
                ens_cls_tokens = self.cls_mlp(ens_cls_tokens)
                ens_dist_tokens = self.dist_mlp(ens_dist_tokens)

            ens_token = (ens_cls_tokens, ens_dist_tokens)
            cls_logits = self.cls_classifier(ens_cls_tokens)
            dist_logits = self.dist_classifier(ens_dist_tokens)
            logits = (cls_logits + dist_logits) / 2

        if distill and self.training and self.teacher_size is not None:
            return ens_token, logits
        else:
            return logits


class MultiViT(nn.Module):
    def __init__(self, model='vit_tiny_patch16_224', drop=0, drop_path=0.1,
                 num_classes_list=[25, 25, 25, 25], num_div=4):
        super().__init__()

        self.model = model
        assert len(num_classes_list) == num_div, 'num of classes is not match num of sub-models'

        self.backbones = nn.ModuleList([])
        for i, num_class in enumerate(num_classes_list):
            self.backbones.append(create_model(model_name=self.model,
                                               num_classes=int(num_class),
                                               drop_rate=drop,
                                               drop_path_rate=drop_path,
                                               drop_block_rate=None, ))
            del self.backbones[i].head
            if 'deit' in self.model:
                del self.backbones[i].head_dist

    def forward(self, x):
        output_features = [model.forward_features(x) for model in self.backbones]
        if 'vit' in self.model:
            output_cls_list = [output_feature['output'] for output_feature in output_features]
            return output_cls_list
        else:
            output_cls_list = [output_feature['output'][0] for output_feature in output_features]
            output_dist_list = [output_feature['output'][1] for output_feature in output_features]
            return output_cls_list, output_dist_list


class EnsMLP(nn.Module):
    def __init__(self, model='vit_tiny_patch16_224', num_class=100, sub_size=192,
                 num_classes_list=[25, 25, 25, 25], teacher_size=None):
        super().__init__()
        self.model = model
        self.sub_size = sub_size
        self.teacher_size = teacher_size
        self.model = model
        self.num_classes = num_class
        self.sum_feature_dim = self.sub_size * len(num_classes_list)

        if self.teacher_size is None:
            self.cls_classifier = nn.Linear(self.sum_feature_dim, self.num_classes)
            if 'deit' in self.model:
                self.dist_classifier = nn.Linear(self.sum_feature_dim, self.num_classes)
        else:
            self.cls_mlp = nn.Linear(self.sum_feature_dim, self.teacher_size)
            self.cls_classifier = nn.Linear(self.teacher_size, self.num_classes)
            if 'deit' in self.model:
                self.dist_mlp = nn.Linear(self.sum_feature_dim, self.teacher_size)
                self.dist_classifier = nn.Linear(self.teacher_size, self.num_classes)

    def forward(self, x, distill=False):
        if 'vit' in self.model:
            ens_cls_tokens = torch.stack(x, 1).view(x[0].shape[0], -1)
            if self.teacher_size is not None:
                ens_cls_tokens = self.cls_mlp(ens_cls_tokens)

            ens_token = ens_cls_tokens
            logits = self.cls_classifier(ens_cls_tokens)

        elif 'deit' in self.model:
            output_cls_list, output_dist_list = x
            ens_cls_tokens = torch.stack(output_cls_list, 1).view(output_cls_list[0].shape[0], -1)
            ens_dist_tokens = torch.stack(output_dist_list, 1).view(output_dist_list[0].shape[0], -1)
            if self.teacher_size is not None:
                ens_cls_tokens = self.cls_mlp(ens_cls_tokens)
                ens_dist_tokens = self.dist_mlp(ens_dist_tokens)

            ens_token = (ens_cls_tokens, ens_dist_tokens)
            cls_logits = self.cls_classifier(ens_cls_tokens)
            dist_logits = self.dist_classifier(ens_dist_tokens)
            logits = (cls_logits + dist_logits) / 2

        if distill and self.training and self.teacher_size is not None:
            return ens_token, logits
        else:
            return logits


class MultiCCT(nn.Module):
    
    def __init__(self, model_type, num_classes_list=[25, 25, 25, 25], num_sub_models=4, input_size=224):
        super().__init__()
        self.model_type = model_type
        assert len(num_classes_list) == num_sub_models, 'num of classes is not match num of sub-models'
        if self.model_type.split('_')[0] == 'cct7':
            from .cct import get_cct7
            cfg = self.model_type.split('_')[-1]
            kernel_size, conv_layers = [int(i) for i in cfg.split('x')]
            self.models = nn.ModuleList(get_cct7(img_size=input_size,
                                                 kernel_size=kernel_size,
                                                 n_conv_layers=conv_layers,
                                                 num_classes=num_classes,
                                                 backbone=True)
                                        for num_classes in num_classes_list)

    def forward(self, x):
        output_features = [model.forward(x)[0] for model in self.models]

        return output_features


class EnsembleCCT(nn.Module):
    def __init__(self, sub_size=256, teacher_size=None, num_sub_models=4, num_classes=100):
        super(EnsembleCCT, self).__init__()
        self.sub_size = sub_size
        self.teacher_size = teacher_size
        self.num_sub_models = num_sub_models
        self.num_classes = num_classes
        self.sum_feature_dim = self.sub_size * self.num_sub_models

        if self.teacher_size is None:
            self.cls_classifier = nn.Linear(self.sum_feature_dim, self.num_classes)
        else:
            self.cls_mlp = nn.Linear(self.sum_feature_dim, self.teacher_size)
            self.cls_classifier = nn.Linear(self.teacher_size, self.num_classes)

    def forward(self, sub_model_features, distill=False):
        '''

        Args:
            sub_model_features (list): last transformer block output feature of sub-models
            distill (bool): whether to distill

        Returns:

        '''
        ens_cls_tokens = torch.stack(sub_model_features, 1).view(sub_model_features[0].shape[0], -1)
        if self.teacher_size is None:
            logits = self.cls_classifier(ens_cls_tokens)
        else:
            cls_token_mlp = self.cls_mlp(ens_cls_tokens)
            logits = self.cls_classifier(cls_token_mlp)

        if distill and self.training:
            return cls_token_mlp, logits
        else:
            return logits