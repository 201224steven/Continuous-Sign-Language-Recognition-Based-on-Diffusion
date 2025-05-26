import torch
import torch.nn as nn
import numpy as np


class Mseloss(nn.Module):

    def __init__(self):
        super(Mseloss, self).__init__()
        self.mseloss = nn.MSELoss(reduction='sum')

    def forward(self, logits, logits_len, label, label_len):
        logits_len = logits_len.numpy()
        label = label.numpy()
        label_len = label_len.numpy()
        label1 = label[:label_len[0]]
        label2 = label[label_len[0]:]
        len1 = (label_len[0] * 2 + 1)
        len2 = (label_len[1] * 2 + 1)
        # index1 = torch.arange(1, len1, 2)
        # index2 = torch.arange(1, len2, 2)

        label1 = label1.reshape((-1, 1))
        label_new1 = np.insert(label1, 0, [0], axis=1)
        label_new1 = label_new1.flatten()
        label_new1 = np.insert(label_new1, len1 - 1, 0)

        label2 = label2.reshape((-1, 1))
        label_new2 = np.insert(label2, 0, [0], axis=1)
        label_new2 = label_new2.flatten()
        label_new2 = np.insert(label_new2, len2 - 1, 0)
        logit1 = logits[:logits_len[0], 0, :]  # logits.shape = [T, B, N]
        logit2 = logits[:logits_len[1], 1, :]  # 因为短的那个输出的补全的部分不是0,与生成的标签计算mseloss时会产生loss值
        map1 = getMap(logits_len[0], len1, label_new1)
        map2 = getMap(logits_len[1], len2, label_new2)
        final_label1 = getLabel(logit1, label_new1, logits_len[0], len1, map1)
        final_label2 = getLabel(logit2, label_new2, logits_len[1], len2, map2)
        final_label1 = torch.from_numpy(final_label1)
        final_label2 = torch.from_numpy(final_label2)
        loss1 = self.mseloss(logit1.cuda(), final_label1.cuda())
        loss2 = self.mseloss(logit2.cuda(), final_label2.cuda())
        loss = (loss1 + loss2) / 2
        return loss


# class Mseloss(nn.Module):
#
#     def __init__(self):
#         super(Mseloss, self).__init__()
#         self.mseloss = nn.MSELoss(reduction='sum')
#
#     def forward(self, logits, logits_len, label, label_len):
#         logits_len = logits_len.numpy()
#         label = label.numpy()
#         label_len = label_len.numpy()
#         T, B, N = logits.size()
#         index = 0
#         loss = 0
#         for i in range(B):
#             if i == B - 1:
#                 mlabel = label[index:]
#             else:
#                 mlabel = label[index:label_len[i]]
#             index += label_len[i]
#             mlen = (label_len[i] * 2 + 1)
#             mlabel = mlabel.reshape((-1, 1))
#             mlabel_new = np.insert(mlabel, 0, [0], axis=1)
#             mlabel_new = mlabel_new.flatten()
#             mlabel_new = np.insert(mlabel_new, mlen - 1, 0)
#             map = getMap(logits_len[i], mlen, mlabel_new)
#             final_label = getLabel(logits[:logits_len[i], i, :], mlabel_new, logits_len[i], mlen, map)
#             final_label = torch.from_numpy(final_label)
#             loss += self.mseloss(logits[:logits_len[i], i, :].cuda(), final_label.cuda())
#         loss = loss / B
#         return loss


def getMap(logit_len, label_len, label):
    map_forward = dp(logit_len, label_len, label, 0)  # label_len行，logit_len列
    map_backward = dp(logit_len, label_len, label, 1)
    map_final = map_forward * map_backward
    # print('fmap', map_final)
    # del map_backward
    # del map_forward
    return map_final


def dp(logit_len, label_len, label, flag):
    map = np.zeros((label_len, logit_len))
    if flag == 0:
        map[0, 0] = 1
        map[1, 0] = 1
        for i in range(1, logit_len):
            for j in range(label_len):
                if label[j] == 0 or (j - 2 >= 0 and label[j] == label[j - 2]):
                    map[j, i] += map[j, i - 1]
                    if j - 1 >= 0:
                        map[j, i] += map[j - 1, i - 1]
                else:
                    map[j, i] += map[j, i - 1]
                    if j - 1 >= 0:
                        map[j, i] += map[j - 1, i - 1]
                        if j - 2 >= 0:
                            map[j, i] += map[j - 2, i - 1]
    else:
        map[label_len - 1, logit_len - 1] = 1
        map[label_len - 2, logit_len - 1] = 1
        for i in range(logit_len - 2, -1, -1):
            for j in range(label_len):
                if label[j] == 0 or (j + 2 < label_len and label[j] == label[j + 2]):
                    map[j, i] += map[j, i + 1]
                    if j + 1 < label_len:
                        map[j, i] += map[j + 1, i + 1]
                else:
                    map[j, i] += map[j, i + 1]
                    if j + 1 < label_len:
                        map[j, i] += map[j + 1, i + 1]
                        if j + 2 < label_len:
                            map[j, i] += map[j + 2, i + 1]
    return map


def getLabel(logit, label, logit_len, label_len, map):
    # final_label = torch.zeros_like(logit)
    # final_label = final_label.cpu().numpy()
    T, N = logit.shape
    final_label = np.zeros((T, N), dtype=np.float32)
    for i in range(logit_len):
        for j in range(label_len):
            if map[j, i] != 0:
                final_label[i, label[j]] = logit[i, label[j]]
    # del T
    # del N
    return final_label


# mseloss = Mseloss()
# logits = torch.randn((10, 2, 16))
# logits_len = torch.Tensor([10, 8])
# label = torch.Tensor([1, 2, 3, 4, 7, 8, 8])
# label_len = torch.Tensor([4, 3])
# loss = mseloss(logits.softmax(-1), logits_len.int(), label.int(), label_len.int())
# print(loss)
