import pdb
import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model,
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            alpha = self.optim_dict['learning_ratio']
            self.optimizer = optim.Adam(
                # [
                #     {'params': model.diffusion_model.parameters()},
                #     {'params': model.crossAttn.parameters(), 'lr': self.optim_dict['base_lr'] * 0.05},
                #     {'params': model.glossEmbedder.parameters(), 'lr': self.optim_dict['base_lr'] * 0.01},  # 注释掉的参数，就相当于被冻结
                #     {'params': model.weights},  # nn.Parameter 的对象本来就是参数，不用在调用 .parameters() 方法
                #     {'params': model.logit_scale},
                #     {'params': model.v2t.parameters()},
                #     {'params': model.t2v.parameters()},
                #     {'params': model.conv2d.parameters(), 'lr': self.optim_dict['base_lr'] * alpha},
                #     {'params': model.conv1d.temporal_conv.parameters(), 'lr': self.optim_dict['base_lr'] * alpha},
                #     {'params': model.temporal_model.parameters(), 'lr': self.optim_dict['base_lr'] * alpha},
                #     {'params': model.classifier.parameters(), 'lr': self.optim_dict['base_lr'] * alpha},
                # ],
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)  # gamma修改为0.5，原为0.2，使用CSL-Daily时修改为0.5
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
