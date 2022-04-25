import copy
import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_
from numba import jit
import pickle


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

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
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

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
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """

    def __init__(self):
        # self.logger = getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError



    def other_parameter(self):
        if hasattr(self, 'other_parameter_name'):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)


class SequentialRecommender(AbstractRecommender):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """
    # type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


class SAS(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, n_node, config):
        super(SAS, self).__init__(n_node, config)

        # load parameters info
        self.n_items = n_node
        self.n_layers = config.num_layer
        self.n_heads = config.num_heads
        self.emb_size = config.hidden_units
        self.hidden_size = config.hidden_units  # same as embedding_size
        self.inner_size = config.inner_units # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.dropout_rate
        self.attn_dropout_prob = config.dropout_rate
        self.hidden_act = config.act
        self.layer_norm_eps = 1e-12
        self.max_seq_length = 300
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        # self.w_1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_1_hot = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        # self.w_1_hot = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.w_2_hot = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1_hot = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2_hot = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w_1_cold = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        # self.w_1_cold = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.w_2_cold = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.glu1_cold = nn.Linear(self.hidden_size, self.hidden_size)
        self.glu2_cold = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.initializer_range = 0.01

        # define layers and loss
        self.embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        # if isinstance(module, (nn.Linear, nn.Embedding)):
        #     # Slightly different from the TF version which uses truncated_normal for initialization
        #     # cf https://github.com/pytorch/pytorch/pull/5617
        #     module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        # elif isinstance(module, nn.LayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            # weight.data.normal_(0, 0.1)
            weight.data.uniform_(-0.1, 0.1)

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
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

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

    def generate_sess_emb_hot(self, item_seq, seq_len, mask):
        seq_h = self.embedding(item_seq)
        hs = torch.sum(seq_h, 1) / seq_len
        len = seq_h.shape[1]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh.float())
        nh = torch.sigmoid(self.glu1_hot(nh) + self.glu2_hot(hs))
        beta = torch.matmul(nh, self.w_2_hot)
        mask = mask.float().unsqueeze(-1)
        sess = beta * mask
        sess_emb = torch.sum(sess * seq_h, 1)
        return sess_emb

    def generate_sess_emb_cold(self, item_seq, seq_len, mask):
        seq_h = self.embedding(item_seq)
        hs = torch.div(torch.sum(seq_h, 1), seq_len)
        len = seq_h.shape[1]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1_cold(nh) + self.glu2_cold(hs))
        beta = torch.matmul(nh, self.w_2_cold)
        mask = mask.float().unsqueeze(-1)
        sess = beta * mask
        sess_emb = torch.sum(sess * seq_h, 1)
        return sess_emb

    def forward(self, item_seq, item_seq_len, mask):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        self.output = self.generate_sess_emb(output, item_seq_len, mask)
        # print(self.output)
        return self.output, self.embedding.weight # [B H]

    def full_sort_predict(self, item_seq, seq_len, mask):
        seq_output, item_emb = self.forward(item_seq, seq_len, mask)
        scores = torch.matmul(seq_output, item_emb.transpose(0, 1))  # [B n_items]
        return scores


def forward(model, i, data):
    session_item, tar, seq_len, mask, hot_sess_items, cold_sess_items, hot_sess_len, \
    cold_sess_len, hot_cold_tar, hot_mask, cold_mask, hot_only_index, cold_only_index = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    seq_len = trans_to_cuda(torch.Tensor(seq_len).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    output, item_emb = model(session_item, seq_len, mask)
    logits = torch.matmul(output, item_emb.transpose(0, 1))
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(logits, tar)

    return tar, loss


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid,score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid,score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids#,k_largest_scores


def train_test(model, train_data, test_data, epoch, opt):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(opt.batch_size)
    for i in slices:
        tar, loss = forward(model, i, train_data)
        # print(loss)
        model.zero_grad()
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    # total_loss = 0
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(opt.batch_size)
    for i in slices:
        session_item, tar, seq_len, mask, hot_sess_items, cold_sess_items, hot_sess_len, \
    cold_sess_len, hot_cold_tar, hot_mask, cold_mask, hot_only_index, cold_only_index = test_data.get_slice(i)
        session_item = trans_to_cuda(torch.Tensor(session_item).long())
        seq_len = trans_to_cuda(torch.Tensor(seq_len).long())
        mask = trans_to_cuda(torch.Tensor(mask).long())
        score = model.full_sort_predict(session_item, seq_len, mask)
        scores = trans_to_cpu(score).detach().numpy()
        index = []
        for idd in range(100):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss


