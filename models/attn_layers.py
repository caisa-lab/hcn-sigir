import torch
import torch.nn as nn
import numpy as np
import math
from geoopt.manifolds.stereographic.manifold import PoincareBall

perterb = 1e-15

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    def single_query_attn_scores(self, key, query):
        euclid_key = key
        euclid_query = query
        scores = torch.bmm(euclid_key, euclid_query.unsqueeze(-1))
        #scores = manifold.mobius_pointwise_mul(g_t,scores)
        denom = torch.norm(euclid_key)
        scores = (1./ denom) * scores
        return scores


    def forward(self, key, query, seq_lens=None):
        scores = self.single_query_attn_scores(key, query)  # shape (batch, seq, 1)
        scaled_scores = torch.nn.functional.softmax(scores, -2)  # softmax on seq dim (shape=same as scors)
        if seq_lens is not None:
            mask = torch.ones_like(scaled_scores).squeeze().type(
                value.dtype).detach()
            for id_in_batch, seq_len in enumerate(seq_lens):
                mask[id_in_batch, seq_len:] = 0.
            scaled_scores = scaled_scores.squeeze() * mask
            # renormalize
            _sums = scaled_scores.sum(-1, keepdim=True)  # sums per row
            scaled_scores = scaled_scores.div(_sums).unsqueeze(-1)
        #scaled_scores = normalize(scaled_scores)
        scaled_scores = scaled_scores + perterb
        #out = manifold.mobius_pointwise_mul(scaled_scores, value)
        return scaled_scores


class HyperAttn(nn.Module):
    def __init__(self):
        super(HyperAttn, self).__init__()
        self.beta = nn.Parameter(torch.Tensor([1.0]),requires_grad=True)
        self.c = nn.Parameter(torch.Tensor([0.0]),requires_grad=True)

    def single_query_attn_scores(self, key, query, g_t, manifold):
        euclid_key = manifold.logmap0(key)
        euclid_query = manifold.logmap0(query).unsqueeze(dim=1)
        scores = manifold.dist(euclid_key,euclid_query)
        #scores = manifold.mobius_pointwise_mul(g_t,scores)
        return self.beta*scores - self.c


    def forward(self, key, query, g_t, manifold):
        """
        Arguments:

            key: Hyperbolic key with shape (batch, seq, hidden_dim)

            query: Hyperbolic query with shape (batch, hidden_dim)

            values: Hyperbolic value with shape (batch, seq, hidden_dim)

            seq_lens: LongTensor of shape (batch,). Used for masking

        Returns:

            Attended value with shape (batch,seq, hidden_dim)

        """
        scores = self.single_query_attn_scores(key, query, g_t, manifold)  # shape (batch, seq, 1)
        scaled_scores = torch.nn.functional.softmax(scores, -2)  # softmax on seq dim (shape=same as scors)
        scaled_scores = scaled_scores + perterb
        #out = manifold.mobius_pointwise_mul(scaled_scores, value)
        return scaled_scores
