import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
from modules.crossAttention import *
from modules.resnet import PositionalEncoding
import modules.resnet as resnet
from modules.DiT import *
from utils.Mseloss import *
from utils.Contrastive_Loss import *

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()

        self.glossEmbedder = GlossEmbedder(num_classes, hidden_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.v2t = MLP(hidden_size)
        self.t2v = MLP(hidden_size)

        self.diffusion_model = DiT(depth=8, hidden_size=1024, num_heads=8, num_classes=num_classes)

        self.pe = PositionalEncoding(d_model=hidden_size)
        self.decoder = CrossAttention_Perciever(dim=hidden_size, num_layer=4)
        self.weights = nn.Parameter(torch.ones(2) / 2, requires_grad=True)

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)

        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x


    def forward(self, x, len_x, diffusion, label=None, label_lgt=None, phase=0):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            framewise = self.conv2d(x.permute(0, 2, 1, 3, 4), len_x).view(batch, temp, -1).permute(0, 2, 1)  # btc -> bct
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(x, lgt)
        visual_cls = tm_outputs['predictions'].mean(dim=0, keepdim=True).transpose(0, 1).squeeze()  # B， D
        visual_cls = self.v2t(visual_cls)

        g, gloss_cls = self.glossEmbedder(label, label_lgt)
        gloss_cls = gloss_cls.squeeze() # B， D
        gloss_cls = self.t2v(gloss_cls)

        # normalized features
        image_features = visual_cls / visual_cls.norm(dim=-1, keepdim=True)
        text_features = gloss_cls / gloss_cls.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T  # B, B
        logits_per_text = logit_scale * text_features @ image_features.T  # B, B

        z = torch.randn_like(g, requires_grad=False).detach()
        sample_outputs, loss_step = diffusion.sample(z, tm_outputs['predictions'].transpose(0, 1), lgt, label_lgt, g)

        tm_feat = tm_outputs['predictions'].permute(1, 0, 2)  # tbc -> btc
        tm_feat  = self.pe(tm_feat)
        predictions, attn = self.deccoder(tm_feat, sample_outputs, sample_outputs, label_lgt, lgt)
        predictions = predictions.permute(1, 0, 2)  # btc -> tbc
        outputs1 = self.classifier(tm_outputs['predictions'])
        outputs2 = self.classifier(predictions)
        outputs = self.weights[0] * outputs1 + self.weights[1] * outputs2

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs["conv_logits"], lgt, batch_first=False, probs=False)

        return {
            "sample_outputs": sample_outputs,
            "loss_step": loss_step,
            "frame_len":len_x,
            "cross_attention": attn,
            "gloss_cls": gloss_cls,
            "visual_cls": visual_cls,
            "visual_vec": logits_per_image, # (B, B)
            "gloss_vec": logits_per_text,  # (B, B)
            "label_len": label_lgt,
            "gloss_embed": g,
            "seq_feat": tm_outputs['predictions'],
            "conv_feat": conv1d_outputs['visual_feat'],
            "feat_len": lgt,
            "conv_logits": conv1d_outputs["conv_logits"],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt, diffusion=None):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss_temp = self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                         label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                         label_lgt.cpu().int()).mean()
                if np.isinf(loss_temp.item()) or np.isnan(loss_temp.item()):
                    continue
                loss += weight * loss_temp
            elif k == 'SeqCTC':
                loss_temp = self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                         label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                         label_lgt.cpu().int()).mean()
                if np.isinf(loss_temp.item()) or np.isnan(loss_temp.item()):
                    continue
                loss += weight * loss_temp
            elif k == 'Dist':
                loss_temp = self.loss['distillation'](ret_dict["conv_logits"],
                                     ret_dict["sequence_logits"].detach(),
                                     use_blank=False)
                if np.isinf(loss_temp.item()) or np.isnan(loss_temp.item()):
                    continue
                loss += weight * loss_temp
            elif k == 'Step':  # RSM_LOSS + GFE_LOSS
                loss_temp = ret_dict["loss_step"]
                if np.isinf(loss_temp.item()) or np.isnan(loss_temp.item()):
                    continue
                loss += weight * loss_temp
            elif k == 'Diff':   # DIFF_LOSS
                loss_temp = diffusion(ret_dict["gloss_embed"], ret_dict["seq_feat"].transpose(0, 1).detach(), ret_dict["feat_len"].cpu().int(),
                                           label_len=ret_dict["label_len"].cpu().int())
                if np.isinf(loss_temp.item()) or np.isnan(loss_temp.item()):
                    continue
                loss += weight * loss_temp
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['l1'] = nn.L1Loss(reduction='mean')
        return self.loss
