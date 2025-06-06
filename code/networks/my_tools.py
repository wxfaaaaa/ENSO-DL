import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class make_embedding(nn.Module):
    def __init__(self, cube_dim, d_size, emb_spatial_size, max_len, device,):
        """
        :param cube_dim: The number of grids in one patch cube
        :param d_size: the embedding length
        :param emb_spatial_size:The number of patches decomposed in a field, S
        :param max_len: look back or prediction length, T
        """
        super().__init__()
        pe = torch.zeros(max_len, d_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        x = torch.arange(0, d_size, 2)
        div_term = torch.pow(10000.0, -x / d_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe_time = pe[None, None]

        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None]
        self.emb_space = nn.Embedding(emb_spatial_size, d_size)

        self.linear = nn.Linear(cube_dim, d_size)
        self.norm = nn.LayerNorm(d_size)

    def forward(self, x):
        assert len(x.size()) == 4
        device = x.device
        embedded_space = self.emb_space(self.spatial_pos.to(device))
        self.pe_time = self.pe_time.to(device)
        x = self.linear(x) + self.pe_time[:, :, : x.size(2)] + embedded_space
        return self.norm(x)

def unfold_func(in_data, kernel_size):
    n_dim = len(in_data.size())
    assert n_dim == 4 or n_dim == 5
    data1 = in_data.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    data1 = data1.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)

    if n_dim == 4:
        data1 = data1.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:
        data1 = data1.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert data1.size(-3) == in_data.size(-3) * kernel_size[0] * kernel_size[1]
    return data1

def fold_func(tensor, output_size, kernel_size):
    tensor = tensor.float()
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 5
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    folded = F.fold(f.flatten(-2),output_size=output_size, kernel_size=kernel_size,stride=kernel_size,)
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), *folded.size()[1:])
    return folded

def T_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    sc = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    sc = sc.to(query.device)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert len(mask.size()) == 2
        if mask.device != query.device:
            mask = mask.to(query.device)
        sc = sc.masked_fill(mask[None, None, None], float("-inf"))
    p_sc = F.softmax(sc, dim=-1)
    if dropout is not None:
        p_sc = dropout(p_sc)
    return torch.matmul(p_sc, value)


def S_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query.transpose(2, 3), key.transpose(2, 3).transpose(-2, -1)) / np.sqrt(d_k)
    p_sc = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_sc = dropout(p_sc)
    return torch.matmul(p_sc, value.transpose(2, 3)).transpose(2, 3)


class make_attention(nn.Module):
    def __init__(self, d_size, nheads, attention_module, dropout):
        super().__init__()
        assert d_size % nheads == 0
        self.d_k = d_size // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_size, d_size) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attention_module = attention_module

    def forward(self, query, key, value, mask=None):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (batch, S, T, d_size)
        Returns:
            (batch, S, T, d_size)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)
        query, key, value = [
            l(x).view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k)
            .permute(0, 3, 1, 2, 4)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x = self.attention_module(query, key, value, mask=mask, dropout=self.dropout)
        x = (x.permute(0, 2, 3, 1, 4).contiguous().view(nbatches, nspace, ntime, self.nheads * self.d_k))
        return self.linears[-1](x)


class miniEncoder(nn.Module):
    def __init__(self, d_size, nheads, dim_feedforward, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_size)
        self.dropout = nn.Dropout(dropout)
        self.time_attn = make_attention(d_size, nheads, T_attention, dropout)
        self.space_attn = make_attention(d_size, nheads, S_attention, dropout)
        self.FC = nn.Sequential(
            nn.Linear(d_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_size),
        )
    def TS_attn(self, query, key, value, mask):
        """
        Returns:
            (batch, S, T, d_size)
        """
        tt = self.time_attn(query, key, value, mask)
        return self.space_attn(tt, tt, tt, mask)

    def forward(self, x, mask=None):
        """
        Inputs：
            [B, patch_num, lb, d_size]
        """
        x1 = self.TS_attn(x, x, x, mask)
        x2 = self.norm(x + self.dropout(x1))
        x3 = self.FC(x2)
        y = self.norm(x2 + self.dropout(x3))
        return y


class miniDecoder(nn.Module):
    def __init__(self, d_size, nheads, dim_feedforward, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_size)
        self.dropout = nn.Dropout(dropout)
        self.encoder_attn = make_attention(d_size, nheads, T_attention, dropout)
        self.time_attn = make_attention(d_size, nheads, T_attention, dropout)
        self.space_attn = make_attention(d_size, nheads, S_attention, dropout)
        self.FC = nn.Sequential(
            nn.Linear(d_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_size),
        )

    def divided_TS_attn(self, query, key, value, mask=None):
        m = self.time_attn(query, key, value, mask)
        return self.space_attn(m, m, m)
    def forward(self, x, en_out, tgt_mask, memory_mask):
        x1 = self.divided_TS_attn(x, x, x,tgt_mask)
        x2 = self.norm(x + self.dropout(x1))
        x3 = self.encoder_attn(x2, en_out, en_out, memory_mask)
        x4 = self.norm(x2 + self.dropout(x3))
        x5 = self.FC(x4)
        y = self.norm(x4 + self.dropout(x5))
        return y

