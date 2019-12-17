# Implementation of QANet based on https://github.com/andy840314/QANet-pytorch-
#
# @article{yu2018qanet,
#   title={Qanet: Combining local convolution with global self-attention for reading comprehension},
#   author={Yu, Adams Wei and Dohan, David and Luong, Minh-Thang and Zhao, Rui and Chen, Kai and Norouzi, Mohammad and Le, Quoc V},
#   journal={arXiv preprint arXiv:1804.09541},
#   year={2018}
# }

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex.configs import AttrDict
from dlex.torch.models.base import BaseModel
from dlex.torch.utils.ops_utils import maybe_cuda
from dlex.torch.utils.variable_length_tensor import get_mask
from torch.nn.modules.activation import MultiheadAttention

from ..datasets import QABatch, QADataset


def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)


class InitializedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias)

        if relu:
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(self.out.weight)
        self.relu = relu

    def forward(self, x):
        return F.relu(self.out(x)) if self.relu else self.out(x)


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + maybe_cuda(signal)).transpose(1, 2)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
            padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size, dropout):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([InitializedConv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([InitializedConv1d(size, size, bias=True) for _ in range(self.n)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = self.dropout(nonlinear)
            x = gate * nonlinear + (1 - gate) * x
        return x


class SelfAttention(nn.Module):
    def __init__(self, connector_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.attn = MultiheadAttention(
            embed_dim=connector_dim,
            num_heads=num_heads,
            dropout=dropout)

    def forward(self, query, mask):
        query = query.permute(2, 0, 1)
        X, _ = self.attn(query, query, query, key_padding_mask=mask)
        return X.permute(1, 2, 0)


class Embedding(nn.Module):
    def __init__(self, connector_dim, word_dim, char_dim, dropout, dropout_char):
        super().__init__()
        self.conv2d = nn.Conv2d(char_dim, connector_dim, kernel_size=(1, 7), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = InitializedConv1d(word_dim + connector_dim, connector_dim, bias=False)
        self.high = Highway(2, connector_dim, dropout)
        self.dropout_word = nn.Dropout(dropout)
        self.dropout_char = nn.Dropout(dropout_char)

    def forward(self, char_emb, word_emb, length):
        char_emb = char_emb.permute(0, 3, 1, 2)
        char_emb = self.dropout_char(char_emb)
        char_emb = self.conv2d(char_emb)
        char_emb = F.relu(char_emb)
        char_emb, _ = torch.max(char_emb, dim=3)
        char_emb = char_emb.squeeze()

        word_emb = self.dropout_word(word_emb)
        word_emb = word_emb.transpose(1, 2)
        emb = torch.cat([char_emb, word_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(
            self,
            conv_num_layers: int,
            conv_num_filters: int,
            kernel_size: int,
            connector_dim: int,
            num_heads: int,
            dropout):
        super().__init__()

        self.convs = nn.ModuleList([DepthwiseSeparableConv(
            conv_num_filters, conv_num_filters, kernel_size
        ) for _ in range(conv_num_layers)])

        self.self_att = SelfAttention(connector_dim, num_heads, dropout)

        self.FFN_1 = InitializedConv1d(conv_num_filters, conv_num_filters, relu=True, bias=True)
        self.FFN_2 = InitializedConv1d(conv_num_filters, conv_num_filters, bias=True)

        self.norm_C = nn.ModuleList([nn.LayerNorm(connector_dim) for _ in range(conv_num_layers)])
        self.norm_1 = nn.LayerNorm(connector_dim)
        self.norm_2 = nn.LayerNorm(connector_dim)

        self.conv_num_layers = conv_num_layers
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, l, num_blocks):
        total_layers = (self.conv_num_layers + 1) * num_blocks
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = self.dropout(out)
            out = conv(out)
            out = self.layer_dropout(out, res, self.dropout_p * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, self.dropout_p * float(l) / total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, self.dropout_p * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return self.dropout(inputs) + residual
        else:
            return inputs + residual


class ContextQueryAttention(nn.Module):
    def __init__(self, connector_dim, dropout):
        super().__init__()
        wC = torch.empty(connector_dim, 1)
        wQ = torch.empty(connector_dim, 1)
        wQC = torch.empty(1, 1, connector_dim)
        bias = torch.empty(1)

        nn.init.xavier_uniform_(wC)
        nn.init.xavier_uniform_(wQ)
        nn.init.xavier_uniform_(wQC)
        nn.init.constant_(bias, 0)

        self.wC = nn.Parameter(wC)
        self.wQ = nn.Parameter(wQ)
        self.wQC = nn.Parameter(wQC)
        self.bias = nn.Parameter(bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, C, Q, maskC, maskQ):
        C = C.transpose(1, 2)  # B * context_max_length * emb_dim
        Q = Q.transpose(1, 2)
        batch_size = C.shape[0]
        S = self.trilinear(C, Q)

        maskC = maskC.view(batch_size, C.shape[1], 1)
        maskQ = maskQ.view(batch_size, 1, Q.shape[1])

        # context-to-query attention
        S1 = F.softmax(mask_logits(S, maskQ), dim=2)
        A = torch.bmm(S1, Q)
        # query-to-context attention
        S2 = F.softmax(mask_logits(S, maskC), dim=1)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)

        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear(self, C, Q):
        """
        Trilinear function (Seo et. al, 2016)
        :param C:
        :param Q:
        :return: similarities between each pair of context and query words
            $f(q, c) = W_0 [q, c, q \odot c]$
        """
        C = self.dropout(C)
        Q = self.dropout(Q)
        res = torch.matmul(C, self.wC).expand([-1, -1, Q.shape[1]]) + \
            torch.matmul(Q, self.wQ).transpose(1, 2).expand([-1, C.shape[1], -1]) + \
            torch.matmul(C * self.wQC, Q.transpose(1, 2)) + \
            self.bias
        return res


class Pointer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = InitializedConv1d(dim * 2, 1)
        self.w2 = InitializedConv1d(dim * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2


class QANet(BaseModel):
    def __init__(self, params: AttrDict, dataset: QADataset):
        super().__init__(params, dataset)
        cfg = self.configs

        # input embedding layer
        self.emb_word = dataset.word_embedding_layer
        self.emb_char = nn.Embedding(len(dataset.vocab_char), cfg.input_embedding.char_dim)
        self.emb = Embedding(cfg.connector_dim, dataset.word_dim, dataset.char_dim, cfg.dropout, cfg.dropout_char)

        # embedding encoder layer
        self.emb_enc = EncoderBlock(
            conv_num_layers=cfg.embedding_encoder.conv_num_layers,
            conv_num_filters=cfg.connector_dim,
            kernel_size=cfg.embedding_encoder.conv_kernel_size,
            connector_dim=cfg.connector_dim,
            num_heads=cfg.embedding_encoder.num_heads or cfg.num_heads,
            dropout=cfg.dropout)

        # context-query attention layer
        self.cq_att = ContextQueryAttention(
            connector_dim=cfg.connector_dim,
            dropout=cfg.dropout)
        self.cq_resizer = InitializedConv1d(cfg.connector_dim * 4, cfg.connector_dim)

        # model encoder layer
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                conv_num_layers=cfg.model_encoder.conv_num_layers,
                conv_num_filters=cfg.connector_dim,
                kernel_size=cfg.model_encoder.conv_kernel_size,
                connector_dim=cfg.connector_dim,
                num_heads=cfg.model_encoder.num_heads or cfg.num_heads,
                dropout=cfg.dropout
            ) for _ in range(cfg.model_encoder.num_blocks)])

        # output layer
        self.out = Pointer(cfg.connector_dim)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, batch):
        cfg = self.params.model
        context_word, context_word_lengths, context_char, question_word, question_word_lengths, question_char = batch.X
        # input embedding layer
        maskC = get_mask(context_word_lengths)
        maskQ = get_mask(question_word_lengths)
        Cw = self.emb_word(context_word)
        Cc = self.emb_char(context_char)
        Qw = self.emb_word(question_word)
        Qc = self.emb_char(question_char)
        C = self.emb(Cc, Cw, Cw.shape[1])
        Q = self.emb(Qc, Qw, Qw.shape[1])

        # embedding encoder layer
        Ce = self.emb_enc(C, ~maskC, 1, 1)
        Qe = self.emb_enc(Q, ~maskQ, 1, 1)

        # context-query attention layer
        X = self.cq_att(Ce, Qe, maskC.float(), maskQ.float())
        M = self.cq_resizer(X)
        M = F.dropout(M, p=cfg.dropout, training=self.training)

        # model encoder layer
        for i, block in enumerate(self.encoder_blocks):
            M = block(M, ~maskC, i * (2 + 2) + 1, cfg.model_encoder.num_blocks)
        M0 = M
        M = self.dropout(M)
        for i, block in enumerate(self.encoder_blocks):
            M = block(M, ~maskC, i * (2 + 2) + 1, cfg.model_encoder.num_blocks)
        M1 = M
        M = self.dropout(M)
        for i, block in enumerate(self.encoder_blocks):
            M = block(M, ~maskC, i * (2 + 2) + 1, cfg.model_encoder.num_blocks)
        M2 = M

        # output layer
        p1, p2 = self.out(M0, M1, M2, maskC.float())
        return p1, p2

    def get_loss(self, batch: QABatch, output):
        p1, p2 = output
        y1, y2 = batch.Y[:, 0], batch.Y[:, 1]
        return F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)

    def infer(self, batch: QABatch):
        p1, p2 = self.forward(batch)
        p1 = torch.argmax(p1, -1).tolist()
        p2 = torch.argmax(p2, -1).tolist()
        y1, y2 = batch.Y[:, 0].tolist(), batch.Y[:, 1].tolist()
        pred = list(map(list, zip(*[p1, p2])))
        ref = list(map(list, zip(*[y1, y2])))
        return pred, ref, None, None
