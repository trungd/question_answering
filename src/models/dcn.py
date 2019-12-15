# Implementations of "Dynamic Coattention Networks For Question Answering"
# Based on: https://github.com/atulkum/co-attention

import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex.configs import AttrDict
from dlex.torch import Batch
from dlex.torch.models.base import BaseModel
from dlex.torch.utils.ops_utils import maybe_cuda
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_lstm_forget_bias(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout, emb_layer=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        # self.embedding = get_pretrained_embedding(emb_matrix)
        self.embedding = emb_layer or nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim, hidden_size, 1, batch_first=True,
            bidirectional=False, dropout=dropout)
        init_lstm_forget_bias(self.encoder)
        self.dropout = nn.Dropout(dropout)
        self.sentinel = nn.Parameter(torch.rand(hidden_size,))

    def forward(self, seq, mask):
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)

        seq_embd = self.embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)
        output, _ = self.encoder(packed)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
        e = self.dropout(e)

        b, _ = list(mask.size())
        # copy sentinel vector at the end
        sentinel_exp = self.sentinel.unsqueeze(0).expand(b, self.hidden_size).unsqueeze(1).contiguous()  # B x 1 x l
        lens = lens.unsqueeze(1).expand(b, self.hidden_size).unsqueeze(1)

        sentinel_zero = maybe_cuda(torch.zeros(b, 1, self.hidden_size))
        e = torch.cat([e, sentinel_zero], 1)  # B x m + 1 x l
        e = e.scatter_(1, lens, sentinel_exp)

        return e


class FusionBiLSTM(nn.Module):
    def __init__(self, hidden_size, dropout_ratio):
        super(FusionBiLSTM, self).__init__()
        self.fusion_bilstm = nn.LSTM(
            3 * hidden_size, hidden_size, 1, batch_first=True,
            bidirectional=True, dropout=dropout_ratio)
        init_lstm_forget_bias(self.fusion_bilstm)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, seq, mask):
        lens = torch.sum(mask, 1)
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        packed = pack_padded_sequence(seq_, lens_sorted, batch_first=True)
        output, _ = self.fusion_bilstm(packed)
        e, _ = pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort)  # B x m x 2l
        e = self.dropout(e)
        return e


class DynamicDecoder(nn.Module):
    def __init__(self, hidden_size, maxout_pool_size, max_dec_steps, dropout):
        super(DynamicDecoder, self).__init__()
        self.max_dec_steps = max_dec_steps
        self.decoder = nn.LSTM(4 * hidden_size, hidden_size, 1, batch_first=True, bidirectional=False)
        init_lstm_forget_bias(self.decoder)

        self.maxout_start = MaxOutHighway(hidden_size, maxout_pool_size, dropout)
        self.maxout_end = MaxOutHighway(hidden_size, maxout_pool_size, dropout)

    def forward(self, U, d_mask, span):
        b, m, _ = U.shape

        curr_mask_s,  curr_mask_e = None, None
        results_mask_s, results_s = [], []
        results_mask_e, results_e = [], []
        step_losses = []

        mask_mult = (1.0 - d_mask.float()) * (-1e30)
        indices = maybe_cuda(torch.arange(0, b))
        s_i = maybe_cuda(torch.zeros(b, dtype=torch.long))
        e_i = maybe_cuda(torch.sum(d_mask, 1) - 1)

        dec_state_i = None
        s_target = None
        e_target = None
        if span is not None:
            s_target = span[:, 0]
            e_target = span[:, 1]
        u_s_i = U[indices, s_i, :]  # (b, 2l)

        for _ in range(self.max_dec_steps):
            u_e_i = U[indices, e_i, :]  # (b, 2l)
            u_cat = torch.cat((u_s_i, u_e_i), 1)  # (b, 4l)

            lstm_out, dec_state_i = self.decoder(u_cat.unsqueeze(1), dec_state_i)
            h_i, c_i = dec_state_i

            s_i, curr_mask_s, step_loss_s = self.maxout_start(
                h_i, U, curr_mask_s, s_i, u_cat, mask_mult, s_target)
            u_s_i = U[indices, s_i, :]  # b x 2l
            u_cat = torch.cat((u_s_i, u_e_i), 1)  # b x 4l

            e_i, curr_mask_e, step_loss_e = self.maxout_end(
                h_i, U, curr_mask_e, e_i, u_cat, mask_mult, e_target)

            if span is not None:
                step_losses.append(step_loss_s + step_loss_e)

            results_mask_s.append(curr_mask_s)
            results_s.append(s_i)
            results_mask_e.append(curr_mask_e)
            results_e.append(e_i)

        result_pos_s = torch.sum(torch.stack(results_mask_s, 1), 1).long()
        result_pos_s = result_pos_s - 1
        p1 = torch.gather(torch.stack(results_s, 1), 1, result_pos_s.unsqueeze(1)).squeeze()

        result_pos_e = torch.sum(torch.stack(results_mask_e, 1), 1).long()
        result_pos_e = result_pos_e - 1
        p2 = torch.gather(torch.stack(results_e, 1), 1, result_pos_e.unsqueeze(1)).squeeze()

        if span is not None:
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / self.max_dec_steps
            loss = torch.mean(batch_avg_loss)
            return loss, p1, p2
        else:
            return None, p1, p2


class MaxOutHighway(nn.Module):
    def __init__(self, hidden_size, maxout_pool_size, dropout):
        super(MaxOutHighway, self).__init__()
        self.hidden_size = hidden_size
        self.maxout_pool_size = maxout_pool_size
        self.linear = nn.Linear(5 * hidden_size, hidden_size, bias=False)
        self.m_t_1_linear = nn.Linear(3 * hidden_size, hidden_size * maxout_pool_size)
        self.m_t_2_linear = nn.Linear(hidden_size, hidden_size * maxout_pool_size)
        self.m_t_12_linear = nn.Linear(2 * hidden_size, maxout_pool_size)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, h_i, U, curr_mask, idx_i_1, u_cat, mask_mult, target=None):
        b, m, _ = U.shape

        r = torch.tanh(self.linear(torch.cat((h_i.view(-1, self.hidden_size), u_cat), 1)))  # (b, l)
        r = self.dropout(r)

        r_expanded = r.unsqueeze(1).expand(b, m, self.hidden_size).contiguous()  # (b, m, l)
        m_t_1 = torch.cat((U, r_expanded), 2).view(-1, 3 * self.hidden_size)  # (b * m, 3l)

        m_t_1 = self.m_t_1_linear(m_t_1)  # (b * m, p * l)
        m_t_1, _ = m_t_1.view(-1, self.hidden_size, self.maxout_pool_size).max(2)  # (b * m, l)

        m_t_2 = self.m_t_2_linear(m_t_1)  # (b * m, p * l)
        m_t_2, _ = m_t_2.view(-1, self.hidden_size, self.maxout_pool_size).max(2)  # (b * m, l)

        alpha_in = torch.cat((m_t_1, m_t_2), 1)  # (b * m, 2l)
        alpha = self.m_t_12_linear(alpha_in)  # (b * m, p)
        alpha, _ = alpha.max(1)  # (b * m)
        alpha = alpha.view(-1, m)  # (b, m)

        alpha = alpha + mask_mult  # b x m
        alpha = F.log_softmax(alpha, 1)  # b x m
        _, idx_i = torch.max(alpha, dim=1)

        if curr_mask is None:
            curr_mask = (idx_i == idx_i)
        else:
            idx_i = idx_i * curr_mask.long()
            idx_i_1 = idx_i_1 * curr_mask.long()
            curr_mask = (idx_i != idx_i_1)

        step_loss = None

        if target is not None:
            step_loss = self.loss(alpha, target)
            step_loss = step_loss * curr_mask.float()

        return idx_i, curr_mask, step_loss


class DCN(BaseModel):
    def __init__(self, params: AttrDict, dataset):
        super().__init__(params, dataset)
        cfg = params.model
        self.hidden_size = cfg.hidden_size

        self.encoder = Encoder(len(dataset.vocab_word), cfg.embedding_dim, cfg.hidden_size, cfg.dropout, dataset.word_embedding_layer)

        self.q_linear = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.fusion_bilstm = FusionBiLSTM(cfg.hidden_size, cfg.dropout)
        self.decoder = DynamicDecoder(
            cfg.hidden_size,
            cfg.maxout_pool_size,
            cfg.max_decoder_steps,
            cfg.dropout)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def get_mask_from_seq_len(self, seq_lens):
        max_len = torch.max(seq_lens)
        indices = maybe_cuda(torch.arange(0, max_len))
        mask = (indices < torch.unsqueeze(seq_lens, 1)).int()
        return mask

    def forward(self, batch: Batch):
        d_seq, d_len, _, q_seq, q_len, _ = batch.X
        q_mask, d_mask = self.get_mask_from_seq_len(q_len), self.get_mask_from_seq_len(d_len)
        span = batch.Y

        D = self.encoder(d_seq, d_mask).transpose(1, 2)  # (B, l, m + 1)

        Q_T = self.encoder(q_seq, q_mask)
        Q = torch.tanh(self.q_linear(Q_T)).transpose(1, 2)  # (B, l, n + 1)

        # co-attention
        D_T = D.transpose(1, 2)  # (B, m + 1, l)
        L = torch.bmm(D_T, Q)  # (B, m + 1, n + 1)
        L_T = L.transpose(1, 2)  # (B, n + 1, m + 1)

        A_Q = F.softmax(L, dim=1)
        C_Q = torch.bmm(D, A_Q)  # (B, l, n + 1)

        A_D = F.softmax(L_T, dim=2)  # (B, n + 1, m + 1)
        C_D = torch.bmm(torch.cat((Q, C_Q), 1), A_D)  # (B, 2l, m + 1)

        C_D_T = C_D.transpose(1, 2)  # (B, m + 1, 2l)

        # BiLSTM
        DC = torch.cat((D_T, C_D_T), 2)  # (B, m + 1, 3l)
        U = self.fusion_bilstm(DC, d_mask)  # [:, :-1, :]  # (B, m, 2l)

        loss, p1, p2 = self.decoder(U, d_mask, span)
        return loss, p1, p2

    def infer(self, batch: Batch):
        _, p1, p2 = self.forward(batch)
        p1 = p1.tolist()
        p2 = p2.tolist()
        y1, y2 = batch.Y[:, 0].tolist(), batch.Y[:, 1].tolist()
        pred = list(map(list, zip(*[p1, p2])))
        ref = list(map(list, zip(*[y1, y2])))
        return pred, ref, None, None

    def get_loss(self, batch, output):
        return output[0]

