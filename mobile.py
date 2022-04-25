import numpy as np
import torch
import copy
import math
import datetime
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_
from numba import jit
import pickle
from typing import Optional,cast
from typing_extensions import Final
from torch import Tensor
from typing import List, Tuple
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(new_x_shape[0],new_x_shape[1], new_x_shape[2], new_x_shape[3])
        return x.permute(0, 2, 1, 3)

    def softmax_x(self, x):
        x_exp = x.exp()
        partition = x_exp.sum(dim=1, keepdim=True)
        return x_exp / partition

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.softmax_x(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.reshape(new_context_layer_shape[0], new_context_layer_shape[1], new_context_layer_shape[2])
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)

        self.dense_2 = nn.Linear(inner_size, inner_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class SASRec(nn.Module):
    def __init__(self, n_node, blocks):
        super(SASRec, self).__init__()

        # load parameters info
        self.n_items = n_node
        self.n_layers = 1
        self.n_heads = 1
        self.hidden_size = 128  # same as embedding_size
        self.emb_size = 128
        self.inner_size = 128 # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = 0.5
        self.attn_dropout_prob = 0.5

        self.layer_norm_eps = 1e-12
        self.max_seq_length = 300
        self.batch_size = 100

        self.initializer_range = 0.01

        # self.embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.layer = TransformerLayer(n_heads=1, hidden_size=128, intermediate_size=128, hidden_dropout_prob=0.5, attn_dropout_prob=0.5, layer_norm_eps=1e-8)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.w = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_1 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.relu = nn.ReLU()
        self.t = 2
        self.p = list(self.parameters())
        self.i = 0

        self.block = blocks  # (12, 38, 88)(8, 4, 4)
        self.block_num = 2
        self.tt_rank = {}
        for i in range(self.block_num):
            if i == 0:
                self.tt_rank[i] = 1
            else:
                self.tt_rank[i] = 100
        self.tt_rank[self.block_num] = 1
        stdv = 1.0 / math.sqrt(self.emb_size)

        dim1 = self.block[0][0]
        dim2 = self.block[1][0]
        self.emb0 = nn.Embedding(self.tt_rank[0]*dim1*dim2, self.tt_rank[1])
        # print(self.emb0)
        dim1 = self.block[0][1]
        dim2 = self.block[1][1]
        self.emb1 = nn.Embedding(int(self.tt_rank[1] / 2), int(dim1 * dim2 * self.tt_rank[2]/2))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        for weight in self.parameters():
            weight.data.uniform_(-0.1, 0.1)  # Tmall

    def generate_sess_emb(self, seq_h, seq_len, mask):
        hs = torch.div(torch.sum(seq_h, 1), seq_len)
        len = seq_h.shape[1]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        mask = mask.float().unsqueeze(-1)
        sess = beta * mask
        sess_emb = torch.sum(sess * seq_h, 1)
        return sess_emb

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask  # fp16 compatibility
        self.i += 1
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def merge_STT(self,emb0, emb1, t):
        N = emb0.shape[1]
        slice = torch.stack(torch.arange(0, N).split(t, dim=-1))
        emb = {}
        for i in range(t):
            emb[i] = torch.mm(torch.index_select(emb0, dim=-1, index=slice[:, i]), emb1)
        embedding = emb[0]
        for i in range(1, t):
            embedding = torch.cat([embedding, emb[i]], dim=-1)

        embedding = embedding.transpose(0, 1)
        slice = []
        for i in range(int(embedding.shape[0]*0.5)):
            slice += [i, i + int(embedding.shape[0]*0.5)]
        # slice = [[i, i + int(embedding.shape[0]/2)] for i in range(int(embedding.shape[0]/2))]
        slice = torch.tensor(slice).long()
        embedding = embedding[slice]
        del slice, emb
        return embedding.transpose(0, 1).reshape(-1, self.emb_size)

    def mapping(self, item_list):
        I = self.block[0]
        L = [1]
        for i in range(1, self.block_num):
            L.append(L[i-1]*I[i])
        # n = range(self.n_items)
        index_list = []
        for i in range(self.block_num - 1, -1, -1):
            index_list.append(torch.true_divide(item_list, L[i]).floor())
            item_list = torch.fmod(item_list, L[i])

        mapped0 = self.emb0.weight.reshape(self.block[0][0], self.block[1][0]*self.tt_rank[1])[index_list[0].long()]
        mapped1 = self.emb1.weight.reshape(int(self.tt_rank[1] / self.t), self.block[0][1], int(self.block[1][1]/self.t)*self.tt_rank[2])[:, index_list[1].long(),:]

        # if self.block_num == 2:
        mapped0 = mapped0.reshape(item_list.shape[0], -1, self.tt_rank[1])
        mapped1 = mapped1.reshape(item_list.shape[0], int(self.tt_rank[1] / self.t), -1)
        mapped_emb = torch.zeros(item_list.shape[0], self.emb_size)
        for i in range(item_list.shape[0]):
            mapped_emb[i] = self.merge_STT(mapped0[i], mapped1[i], self.t)
        return mapped_emb

    def forward(self, item_seq, item_seq_len, mask, item_list):
        position_ids = torch.arange(item_seq.size(1))
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        embedding = self.mapping(item_list)
        zero = torch.zeros(1, self.emb_size)
        embedding = torch.cat([zero, embedding], dim=0)
        get = lambda i: embedding[item_seq[i]]
        # seq_h = torch.cuda.FloatTensor(self.size, list(item_seq.shape)[1], self.emb_size).fill_(0)
        seq_h = torch.zeros(self.batch_size, list(item_seq.shape)[1], self.emb_size)
        for i in torch.arange(item_seq.shape[0]):
            seq_h[i] = get(i)
        item_emb = seq_h
        item_emb = item_emb.reshape(self.batch_size, -1, self.emb_size)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.layer(input_emb, extended_attention_mask)
        output = trm_output
        self.output = self.generate_sess_emb(output, item_seq_len, mask)

        return self.output, embedding


class Data():
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0])
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        mask = []
        tar = self.targets[index]
        item_set = set()
        item_dict = {}
        cnt = 1
        for t, session in enumerate(inp):
            item_set.update(session)
            for i in session:
                if i not in item_dict:
                    item_dict[i] = cnt
                    cnt += 1
        item_set = list(item_set)
        for t, session in enumerate(inp):
            nonzero_elems = np.nonzero(session)[0]
            length = len(nonzero_elems)
            batch_item = []
            for cnt, item in enumerate(session):
                batch_item.append(item_dict[item])
            sess_items = batch_item + (max_n_node - length) * [0]
            session_len.append([length])
            items.append(sess_items)
            mask.append([1] * length + (max_n_node - length) * [0])
        return items, tar, session_len, mask, list(item_set)


dataset = "Tmall"
train_data = pickle.load(open('../datasets/' + dataset + '/train.txt', 'rb'))
if dataset == 'diginetica':
    n_node = 43097 + 3
elif dataset == 'Tmall':
    n_node = 40727 + 2
elif dataset == 'retailrocket':
    n_node = 36968 + 4
else:
    n_node = 309 + 1
train_data = Data(train_data, shuffle=False)
blocks = [[169, 241], [16, 8]]
model = SASRec(n_node, blocks)
print('start training: ', datetime.datetime.now())
total_loss = 0.0
slices = train_data.generate_batch(100)
for i in slices:
    session_item, tar, seq_len, mask, item_list = train_data.get_slice(i)
    session_item = torch.Tensor(session_item).long()
    seq_len = torch.Tensor(seq_len).long()
    mask = torch.Tensor(mask).long()
    item_list = torch.Tensor(item_list).long()
    sess_rep, item_emb = model(session_item, seq_len, mask, item_list)
    embedding = model.merge_STT(model.emb0.weight, model.emb1.weight, model.t)
    logits = torch.matmul(sess_rep, embedding.transpose(0, 1))
    loss_func = nn.CrossEntropyLoss()
    tar = torch.Tensor(tar).long()
    loss = loss_func(logits, tar)
    model.zero_grad()
    loss.backward()
    model.optimizer.step()
    total_loss += loss.item()
    torch.save(model.state_dict(), "sas.pt")
# #
model = SASRec(n_node, blocks)
test_data = pickle.load(open('../datasets/' + dataset + '/test.txt', 'rb'))
test_data = Data(test_data, shuffle=False)
slices = test_data.generate_batch(100)
checkpoint = torch.load("sas.pt", map_location=torch.device('cpu'))
model.eval()
# quantized_decoder = torch.quantization.quantize_dynamic(decoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
session_item, tar, seq_len, mask, item_list = test_data.get_slice(slices[-5])
print(len(session_item), len(session_item[0]), len(item_list), len(seq_len))
session_item = torch.Tensor(session_item).long()
seq_len = torch.Tensor(seq_len).long()
mask = torch.Tensor(mask).long()
item_list = torch.Tensor(item_list).long()
traced_encoder = torch.jit.trace(model, (session_item, seq_len, mask, item_list))

from torch.utils.mobile_optimizer import optimize_for_mobile

traced_encoder_optimized = optimize_for_mobile(traced_encoder)
traced_encoder_optimized._save_for_lite_interpreter(r"/Users/xiaxin/AndroidStudioProjects/Seq2SeqNMT/app/src/main/assets/sas_.ptl")
#
