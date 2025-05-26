import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # self.layeNorm = nn.LayerNorm(dim)

    def forward(self, queries, keys, values, len_x, lgt):
        b, n, _, h = *queries.shape, self.heads
        b2, n2, _ = keys.shape

        # 分到两张GPU上训练，相当于batch_size=1, 所以不用mask

        # maskK = get_mask2(n2, len_x)
        # mask = maskK.unsqueeze(1).unsqueeze(1).cuda()
        # mask = mask == 0
        # print(mask.shape)
        # 要mask的地方区乘负无穷，这样softmax之后才是0

        # queries = self.layeNorm(queries)
        # keys = self.layeNorm(keys)
        # values = self.layeNorm(values)

        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        queries = queries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b2, n2, h, -1).transpose(1, 2)
        values = values.view(b2, n2, h, -1).transpose(1, 2)

        dots = torch.einsum('bhid, bhjd->bhij', queries, keys) * self.scale
        if b == 2:
            maskK = get_mask2(n2, len_x)
            mask = maskK.unsqueeze(1).unsqueeze(1).cuda()
            mask = mask == 0
            dots = dots.masked_fill(mask, float("-inf"))
        # dots = dots.masked_fill(mask, float("-inf")) # 进行 mask 操作， mask 掉序列中 padding 部分, masked_fill 将 True 的位置进行替换， masked_fill 可以使用 2*1*1*j 的mask
        attn = dots.softmax(dim=-1)
        # torch.save(attn, 'similarity.pt')

        out = torch.einsum('bhij, bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        attn2 = attn.mean(dim=1)
        return out, attn2


class CrossAttention_Perciever(nn.Module):
    def __init__(self, dim, num_layer):
        super().__init__()
        self.ca = CrossAttention(dim)
        # layerNorm = nn.LayerNorm(dim)
        transformerEncodeLayer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True)
        self.transformerEncoder = nn.TransformerEncoder(transformerEncodeLayer, num_layers=num_layer)

    def forward(self, queries, keys, values, len_x, lgt):
        b, n, _ = queries.shape
        crossAtten, attn = self.ca(queries, keys, values, len_x, lgt)

        maxlen = lgt[0] if lgt[0] > lgt[1] else lgt[1]
        maxlen = int(maxlen)
        mask = get_mask(maxlen, lgt)
        # 3 mask = torch.concat([mask1, mask2], dim=0)  # mask.shape = B,L
        mask = mask.cuda()

        # crossAtten = crossAtten + queries  # 残差连接

        out = self.transformerEncoder(crossAtten, src_key_padding_mask=mask)  # 只用对 key 进行mask就可以了，因为最后解码的时候，传了lgt这个参数，会忽略掉最后的那些padding的序列
        return out, attn


class TransEncoder(nn.Module):
    def __init__(self, dim, layers):
        super().__init__()
        transformerEncodeLayer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.transformerEncoder = nn.TransformerEncoder(transformerEncodeLayer, num_layers=layers)
    def forward(self, x, lgt):
        n, b, _ = x.shape

        mask = get_mask(n, lgt)
        mask = mask.cuda()

        out = self.transformerEncoder(x, src_key_padding_mask=mask)
        return out



def get_mask(seq_len, len):
    # print(len.shape)
    n = list(len.shape)[0]
    mask = torch.empty((1, seq_len))
    for i in range(n):
        mask_temp = torch.ones(seq_len) == 1
        for k in range(int(len[i])):
            mask_temp[k] = False
        mask_temp = mask_temp.unsqueeze(0)
        mask = torch.concat([mask, mask_temp], dim=0)
    mask = mask[1:, :]
    return mask

def get_mask2(seq_len, len):
    # print(len.shape)
    n = list(len.shape)[0]
    mask = torch.empty((1, seq_len))
    # print(seq_len, len[0])
    for i in range(n):
        mask_temp = torch.zeros(seq_len)
        for k in range(int(len[i])):
            mask_temp[k] = 1
        mask_temp = mask_temp.unsqueeze(0)
        mask = torch.concat([mask, mask_temp], dim=0)
    mask = mask[1:, :]
    return mask

def get_mask3(seq_len, len):
    # print(len.shape)
    n = list(len.shape)[0]
    mask = torch.empty((1, seq_len))
    for i in range(n):
        mask_temp = torch.zeros(seq_len)
        for k in range(int(len[i])):
            mask_temp[k] = 1
        mask_temp = mask_temp.unsqueeze(0)
        mask = torch.concat([mask, mask_temp], dim=0)
    mask = mask[1:, :]
    return mask
