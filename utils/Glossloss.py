import torch
import torch.nn as nn
import numpy as np


class Glossloss(nn.Module):

    def __init__(self):
        super(Glossloss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x, g):
        matrix = torch.matmul(x, g.transpose(1, 2)).cuda()
        index = torch.max(matrix, 1)[1].cuda()
        label = torch.zeros(matrix.size()).cuda()
        for i in range(matrix.size(0)):
            label[i][index] = 1
        loss = self.celoss(matrix, label)
        return loss