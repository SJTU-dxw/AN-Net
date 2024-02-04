import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
import scipy.stats as st


class TrafficDataset(Dataset):
    def __init__(self, data, lb):
        self.data = data
        self.lb = lb

    def __len__(self):
        return len(self.lb)

    def __getitem__(self, index):
        data = self.data[index]
        return data, self.lb[index]


class TrafficTripleDataset(Dataset):
    def __init__(self, data, seg, lb):
        self.data = data
        self.seg = seg
        self.lb = lb

    def __len__(self):
        return len(self.lb)

    def __getitem__(self, index):
        data = self.data[index]
        seg = self.seg[index]
        return (data, seg), self.lb[index]


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout, dropatt, drop_path_rate):
        super(MultiheadAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.dropout = dropout
        self.dropatt = dropatt

        self.att_drop = nn.Dropout(self.dropatt)
        self.hid_drop = nn.Dropout(self.dropout)

        self.q_head = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_head = nn.Linear(d_model, n_head * d_head)
        self.v_head = nn.Linear(d_model, n_head * d_head)

        self.post_proj = nn.Linear(n_head * d_head, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1. / np.sqrt(d_head)

        if drop_path_rate == 0:
            self.drop_path = nn.Identity()
        else:
            self.drop_path = DropPath(drop_path_rate)

    def forward(self, q, k, v):
        q_head = self.q_head(q).transpose(0, 1)
        k_head = self.k_head(k).transpose(0, 1)
        v_head = self.v_head(v).transpose(0, 1)

        q_head = q_head * self.scale
        # content based attention score
        content_score = torch.einsum("...ind,...jnd->...nij", q_head, k_head)
        # merge attention scores
        attn_score = content_score

        # precision safe
        dtype = attn_score.dtype
        attn_score = attn_score.float()

        # attention probability
        attn_prob = torch.softmax(attn_score, dim=-1)
        attn_prob = attn_prob.type(dtype)

        attn_prob = self.att_drop(attn_prob)
        # attention output
        attn_vec = torch.einsum("...nij,...jnd->...ind", attn_prob, v_head)

        attn_out = self.post_proj(attn_vec)
        attn_out = self.hid_drop(attn_out).transpose(0, 1)

        output = self.layer_norm(q + self.drop_path(attn_out))
        return output


def RE(x, training):
    if not training:
        x = torch.cat(x, dim=1)
        x = torch.mean(x, dim=1)
        return x
    x_all = []
    sum_all = []
    for x0 in x:
        shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        random_tensor = x0.new_empty(shape).bernoulli_(0.8)
        random_tensor *= torch.from_numpy(np.random.beta(4, 4, size=shape)).cuda().to(torch.float32)
        sum_all.append(random_tensor)
        x_all.append(x0 * random_tensor)
    sum_all = torch.cat(sum_all, dim=1).sum(1)
    x_all = torch.cat(x_all, dim=1).sum(1)
    x_out = x_all / (sum_all + 1e-5)
    return x_out


class SingleModel(nn.Module):
    def __init__(self, num_layers, with_relu, with_ht, num_class, temp):
        super().__init__()
        self.vocab_size = 10000
        self.embedding = nn.Embedding(self.vocab_size, 128)
        self.encoder = nn.GRU(128, 128, num_layers=2, bidirectional=True, batch_first=True)

        self.fc = []
        for i in range(num_layers):
            if i == 0:
                self.fc.append(nn.Linear(512, 256))
            else:
                self.fc.append(nn.Linear(256, 256))
            if with_relu and i != num_layers - 1:
                self.fc.append(nn.ReLU())
        self.fc = nn.Sequential(*self.fc)

        self.transformer_layer = TransLayer(with_ht, temp)

        self.classifier_fc = nn.Linear(256, num_class)

    def forward(self, x):
        input_sequence = torch.clamp(x, max=self.vocab_size-1, min=0).long()
        B = input_sequence.shape[0]
        x = self.embedding(input_sequence)
        x = x.view(B, 10, 10, 128)
        x = x.view(B * 10, 10, 128)

        encoder_out = self.encoder(x)[1].transpose(0, 1).flatten(1)
        encoder_out = encoder_out.view(B, 10, 512)
        encoder_out = self.fc(encoder_out)

        out = self.transformer_layer(encoder_out)
        out = torch.mean(out, dim=1)

        return out.unsqueeze(1), self.classifier_fc(out)


def extract_statistical(arr):
    statistical_arr = []

    statistical_arr.append(np.expand_dims(np.mean(arr, axis=-1), axis=-1))
    statistical_arr.append(np.expand_dims(np.std(arr, axis=-1), axis=-1))
    statistical_arr.append(np.expand_dims(st.skew(arr, axis=-1), axis=-1))
    statistical_arr.append(np.expand_dims(st.kurtosis(arr, axis=-1), axis=-1))
    statistical_arr.append(np.expand_dims(np.median(arr, axis=-1), axis=-1))
    statistical_arr.append(np.expand_dims(np.min(arr, axis=-1), axis=-1))
    statistical_arr.append(np.expand_dims(np.max(arr, axis=-1), axis=-1))

    statistical_arr = np.concatenate(statistical_arr, axis=-1)
    return statistical_arr


class StatisticModel(nn.Module):
    def __init__(self, num_layers, with_relu, with_ht, num_class, temp):
        super().__init__()
        self.bn = nn.BatchNorm1d(13)
        self.fc = []
        for i in range(num_layers):
            if i == 0:
                self.fc.append(nn.Linear(13, 256))
            else:
                self.fc.append(nn.Linear(256, 256))
            if with_relu and i != num_layers - 1:
                self.fc.append(nn.ReLU())
        self.fc = nn.Sequential(*self.fc)

        self.transformer_layer = TransLayer(with_ht, temp)
        self.classifier_fc = nn.Linear(256, num_class)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 10, 10)
        stat = extract_statistical(x.cpu().detach().numpy())
        spectral_arr = np.fft.fft(x.cpu().detach().numpy(), axis=-1)
        spectral_arr = np.abs(spectral_arr)
        spec = spectral_arr[:, :, :6]
        x = torch.from_numpy(np.nan_to_num(np.concatenate([stat, spec], axis=-1))).cuda().to(torch.float32)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)

        out = self.fc(x)
        out = self.transformer_layer(out)
        out = torch.mean(out, dim=1)

        return out.unsqueeze(1), self.classifier_fc(out)


class Payload_Extractor(nn.Module):
    def __init__(self, num_layers, with_relu, with_ht, num_class, temp):
        super(Payload_Extractor, self).__init__()
        self.lstm = nn.GRU(64, 128, num_layers=2, batch_first=True)
        self.fc = []
        for i in range(num_layers):
            if i == 0:
                self.fc.append(nn.Linear(256, 256))
            else:
                self.fc.append(nn.Linear(256, 256))
            if with_relu and i != num_layers - 1:
                self.fc.append(nn.ReLU())
        self.fc = nn.Sequential(*self.fc)

        self.transformer_layer = TransLayer(with_ht, temp)
        self.classifier_fc = nn.Linear(256, num_class)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 10, 10, 64)
        x = x.view(B * 10, 10, 64)

        x = self.lstm(x)[1].transpose(0, 1).flatten(1)
        x = x.view(B, 10, 256)

        out = self.fc(x)
        out = self.transformer_layer(out)
        out = torch.mean(out, dim=1)

        return out.unsqueeze(1), self.classifier_fc(out)


class FinalModel(nn.Module):
    def __init__(self, num_class, num_layers, with_relu, with_ht, with_re, dataset, temp):
        super(FinalModel, self).__init__()
        self.Payload_Model = Payload_Extractor(num_layers, with_relu, with_ht, num_class, temp)

        self.Length_Model = StatisticModel(num_layers, with_relu, with_ht, num_class, temp)
        self.Length_Model_2 = SingleModel(num_layers, with_relu, with_ht, num_class, temp)

        self.IAT_Model = StatisticModel(num_layers, with_relu, with_ht, num_class, temp)

        self.TTL_Model = StatisticModel(num_layers, with_relu, with_ht, num_class, temp)
        self.TTL_Model_2 = SingleModel(num_layers, with_relu, with_ht, num_class, temp)

        self.IPFlag_Model = StatisticModel(num_layers, with_relu, with_ht, num_class, temp)
        self.IPFlag_Model_2 = SingleModel(num_layers, with_relu, with_ht, num_class, temp)

        self.TCPFlag_Model = StatisticModel(num_layers, with_relu, with_ht, num_class, temp)
        self.TCPFlag_Model_2 = SingleModel(num_layers, with_relu, with_ht, num_class, temp)

        self.fc = nn.Linear(256, num_class)
        self.with_re = with_re
        self.dataset = dataset

    def split(self, x):
        x_Payload = x[:, :, 5:]
        x = x[:, :, :5]
        x_Length = x[:, :, 0]
        x_IAT = x[:, :, 1]
        x_TTL = x[:, :, 2]
        x_IPFlag = x[:, :, 3]
        x_TCPFlag = x[:, :, 4]

        return x_Payload, x_Length, x_IAT, x_TTL, x_IPFlag, x_TCPFlag

    def forward(self, x):
        x_Payload, x_Length, x_IAT, x_TTL, x_IPFlag, x_TCPFlag = self.split(x)
        if self.dataset == 0:
            Length_feature, _ = self.Length_Model(x_Length)
            IAT_feature, _ = self.IAT_Model(x_IAT)
            TTL_feature, _ = self.TTL_Model(x_TTL)
            TCP_feature, _ = self.TCPFlag_Model(x_TCPFlag)
            result = [Length_feature, IAT_feature, TTL_feature, TCP_feature]
        if self.dataset == 1:
            Payload_feature, _ = self.Payload_Model(x_Payload)
            Length_feature, _ = self.Length_Model_2(x_Length)
            IAT_feature, _ = self.IAT_Model(x_IAT)
            TTL_feature, _ = self.TTL_Model_2(x_TTL)
            result = [Payload_feature, Length_feature, IAT_feature, TTL_feature]
        if self.dataset == 2:
            Payload_feature, _ = self.Payload_Model(x_Payload)
            Length_feature, _ = self.Length_Model_2(x_Length)
            IAT_feature, _ = self.IAT_Model(x_IAT)
            IP_feature, _ = self.IPFlag_Model_2(x_IPFlag)
            TCP_feature, _ = self.TCPFlag_Model_2(x_TCPFlag)
            result = [Payload_feature, Length_feature, IAT_feature, IP_feature]
        if self.dataset == 3:
            Payload_feature, _ = self.Payload_Model(x_Payload)
            Length_feature, _ = self.Length_Model_2(x_Length)
            IAT_feature, _ = self.IAT_Model(x_IAT)
            IP_feature, _ = self.IPFlag_Model_2(x_IPFlag)
            TCP_feature, _ = self.TCPFlag_Model_2(x_TCPFlag)
            result = [Payload_feature, Length_feature, IAT_feature, IP_feature]
        if self.dataset == 5:
            Payload_feature, _ = self.Payload_Model(x_Payload)
            Length_feature, _ = self.Length_Model_2(x_Length)
            IAT_feature, _ = self.IAT_Model(x_IAT)
            IP_feature, _ = self.IPFlag_Model_2(x_IPFlag)
            TCP_feature, _ = self.TCPFlag_Model_2(x_TCPFlag)
            result = [Payload_feature, Length_feature, IAT_feature, IP_feature]

        if self.with_re:
            x = RE(result, self.training)
        else:
            x = torch.cat(result, dim=1)
            x = torch.mean(x, dim=1)

        out = self.fc(x)
        return out


class FSNet(nn.Module):
    def __init__(self, num_class):
        super(FSNet, self).__init__()
        vocab_size = 10000
        self.embedding = nn.Embedding(vocab_size, 128)
        self.encoder = nn.GRU(128, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.decoder = nn.GRU(512, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024, num_class)
        self.fc_reconstruction = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = torch.clamp(x, max=9999)
        x = self.embedding(x.long())
        encoder_out = self.encoder(x)[1].transpose(0, 1).flatten(1)
        x = encoder_out.unsqueeze(1)
        x = x.repeat(1, 100, 1)
        reconstruction_out, decoder_out = self.decoder(x)
        decoder_out = decoder_out.transpose(0, 1).flatten(1)

        out = torch.cat([encoder_out, decoder_out], dim=-1)
        out = self.fc(out)

        reconstruction_out = self.fc_reconstruction(reconstruction_out)
        return out, reconstruction_out


class MulHeadAttention(nn.Module):
    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale=True, temp=1.0):
        super(MulHeadAttention, self).__init__()
        self.heads_num = heads_num
        self.temp = temp

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
        )

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, position_bias=None):
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, self.inner_hidden_size)

        query, key, value = [l(x). \
                                 view(batch_size, -1, heads_num, per_head_size). \
                                 transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                             ]

        if self.with_scale:
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
            scores = torch.matmul(query, key.transpose(-2, -1))
            scores = scores / self.temp
        else:
            scores = torch.matmul(query, key.transpose(-2, -1))
            scores = scores / math.sqrt(float(per_head_size))

        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output


class TransLayer(nn.Module):
    def __init__(self, with_ht, temp):
        super(TransLayer, self).__init__()
        self.self_attn = MulHeadAttention(
            256, 4, 64, 0.1, has_bias=True, with_scale=with_ht, temp=temp
        )
        self.dropout_1 = nn.Dropout(0.1)
        self.feed_forward = PositionwiseFeedForward(
            256, 512, True
        )
        self.dropout_2 = nn.Dropout(0.1)

        self.layer_norm_1 = LayerNorm(256)
        self.layer_norm_2 = LayerNorm(256)

    def forward(self, hidden, position_bias=None):
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, position_bias))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)
        return output


class AttnLSTM(nn.Module):
    def __init__(self, num_class):
        super(AttnLSTM, self).__init__()
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 1, bias=False)
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):
        x = x[:, :20, :]
        x = self.lstm(x)[0]
        u = self.tanh(self.fc1(x))
        score = self.fc2(u).squeeze(-1)
        score = F.softmax(score, dim=-1).unsqueeze(-1)

        c = x * score
        c = torch.sum(c, dim=1)

        out = self.fc(c)
        return out


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        hidden_states = self.gamma * (x - mean) / (std + self.eps)

        return hidden_states + self.beta


class WordPosSegEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """

    def __init__(self, vocab_size):
        super(WordPosSegEmbedding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.max_seq_length = 512
        self.word_embedding = nn.Embedding(vocab_size, 768)
        self.position_embedding = nn.Embedding(self.max_seq_length, 768)
        self.segment_embedding = nn.Embedding(3, 768)
        self.layer_norm = LayerNorm(768)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(
            torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long)
                .unsqueeze(0)
                .repeat(word_emb.size(0), 1)
        )
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)
        return emb


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale=True):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
        )

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, mask, position_bias=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, self.inner_hidden_size)

        query, key, value = [l(x). \
                                 view(batch_size, -1, heads_num, per_head_size). \
                                 transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                             ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer. """

    def __init__(self, hidden_size, feedforward_size, has_bias=True):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size, bias=has_bias)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size, bias=has_bias)
        self.act = gelu

    def forward(self, x):
        inter = self.act(self.linear_1(x))
        output = self.linear_2(inter)
        return output


class TransformerLayer(nn.Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()
        # Multi-headed self-attention.
        self.self_attn = MultiHeadedAttention(
            768, 12, 64, 0.1, has_bias=True, with_scale=True
        )
        self.dropout_1 = nn.Dropout(0.1)

        # Feed forward layer.
        self.feed_forward = PositionwiseFeedForward(
            768, 3072, True
        )
        self.dropout_2 = nn.Dropout(0.1)

        self.layer_norm_1 = LayerNorm(768)
        self.layer_norm_2 = LayerNorm(768)

    def forward(self, hidden, mask, position_bias=None):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask, position_bias))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)
        return output


class TransformerEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.layers_num = 12
        self.transformer = nn.ModuleList(
            [TransformerLayer() for _ in range(self.layers_num)]
        )

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = emb.size()

        mask = (seg > 0). \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        hidden = emb

        position_bias = None

        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask, position_bias=position_bias)

        return hidden


class BertModel(nn.Module):
    def __init__(self, num_class, vocab_size):
        super(BertModel, self).__init__()
        self.embedding = WordPosSegEmbedding(vocab_size)
        self.encoder = TransformerEncoder()
        self.output_layer_1 = nn.Linear(768, 768)
        self.output_layer_2 = nn.Linear(768, num_class)

    def forward(self, src, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        src = src.long()
        seg = seg.long()
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)

        return logits
