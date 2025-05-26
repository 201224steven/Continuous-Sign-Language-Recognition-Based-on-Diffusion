import numpy
import torch
import torch.nn as nn
import numpy as np

class Ctloss(nn.Module):
    def __init__(self):
        super(Ctloss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, attn, len_x, lgt):
        len_x = len_x.numpy()
        lgt = lgt.numpy()
        attn1 = attn[0]
        attn2 = attn[1]
        attn1 = attn1[:lgt[0], :len_x[0]]
        attn2 = attn2[:lgt[1], :len_x[1]]
        ct_label1 = getLabel(attn1)
        ct_label2 = getLabel(attn2)
        # ct_label1 = torch.from_numpy(ct_label1)
        # ct_label2 = torch.from_numpy(ct_label2)
        attn1_torch = attn[0, :lgt[0], :len_x[0]]
        attn2_torch = attn[1, :lgt[1], :len_x[1]]
        loss1 = self.celoss(attn1_torch.cuda(), ct_label1.cuda())
        loss2 = self.celoss(attn2_torch.cuda(), ct_label2.cuda())
        loss = (loss1 + loss2) / 2
        return loss



def getLabel(attn):
    n1, n2 = attn.shape
    label = torch.zeros_like(attn)
    idx = torch.argsort(attn, axis=1)
    for i in range(n1):
        idx_row = idx[i][-1:]  # -n 表示取相似度前 n 大的 frame-timestep对作为正样本
        label[i][idx_row] = 1
    return label

# ca = torch.randn((2, 10, 15))
# len_x = torch.Tensor((15, 12)).int()
# lgt = torch.Tensor((10, 8)).int()
# ctloss = Ctloss()
# loss = ctloss(ca, len_x, lgt)
# print(loss)