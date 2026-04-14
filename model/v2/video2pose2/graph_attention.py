import torch.nn as nn
import torch


class GraphMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nheads=4,
        dropout=0.1,
        max_path_len=5,
        value_emb=False,
        use_tree_mask=False,
    ):
        super().__init__()
        assert d_model % nheads == 0

        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(d_model, nheads * self.att_size)
        self.linear_k = nn.Linear(d_model, nheads * self.att_size)
        self.linear_v = nn.Linear(d_model, nheads * self.att_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(nheads * self.att_size, d_model)

        self.max_path_len = max_path_len
        self.value_emb_flag = value_emb
        self.use_tree_mask = use_tree_mask

        self.topology_key_emb = nn.Embedding(max_path_len + 1, d_model)
        self.edge_key_emb = nn.Embedding(6, d_model)
        self.topology_query_emb = nn.Embedding(max_path_len + 1, d_model)
        self.edge_query_emb = nn.Embedding(6, d_model)

        if value_emb:
            self.topology_value_emb = nn.Embedding(max_path_len + 1, d_model)
            self.edge_value_emb = nn.Embedding(6, d_model)

    def forward(
        self,
        q,               # [B,J,D]
        k,               # [B,J,D]
        v,               # [B,J,D]
        distance,        # [B,J,J] or [B,F,J,J]
        edge_attr,       # [B,J,J] or [B,F,J,J]
        mask=None,       # [B,J] or [B,F,J]
        tree_mask=None,  # [B,J,J] or [B,F,J,J]
    ):
        if distance.dim() == 4:
            B, F_, J, _ = distance.shape
            distance = distance.reshape(B * F_, J, J)
            edge_attr = edge_attr.reshape(B * F_, J, J)

            if mask is not None:
                mask = mask.reshape(B * F_, J)

            if tree_mask is not None:
                tree_mask = tree_mask.reshape(B * F_, J, J)

        orig_q_size = q.size()
        batch_size = q.size(0)
        d_k = self.att_size
        d_v = self.att_size

        q = self.linear_q(q).view(batch_size, -1, self.nheads, d_k).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, -1, self.nheads, d_k).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.nheads, d_v).transpose(1, 2)

        seq_len = v.shape[2]
        num_hop_types = self.max_path_len + 1
        num_edge_types = 6

        query_hop_emb = self.topology_query_emb.weight.view(1, num_hop_types, self.nheads, d_k).transpose(1, 2)
        query_edge_emb = self.edge_query_emb.weight.view(1, num_edge_types, self.nheads, d_k).transpose(1, 2)
        key_hop_emb = self.topology_key_emb.weight.view(1, num_hop_types, self.nheads, d_k).transpose(1, 2)
        key_edge_emb = self.edge_key_emb.weight.view(1, num_edge_types, self.nheads, d_k).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(query_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1))

        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1))

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(key_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1))

        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1))

        spatial_bias = query_hop + key_hop
        edge_bias = query_edge + key_edge

        attn_score = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bias
        attn_score = attn_score * self.scale

        # tree mask
        if tree_mask is not None and self.use_tree_mask:
            attn_score = attn_score.masked_fill(~tree_mask[:, None, :, :], float("-inf"))

        # key mask
        if mask is not None:
            mask_k = mask[:, None, None, :]
            attn_score = attn_score.masked_fill(~mask_k, float("-inf"))

        # 避免整行都是 -inf 导致 NaN
        invalid_rows = torch.isinf(attn_score).all(dim=-1, keepdim=True)
        attn_score = torch.where(invalid_rows, torch.zeros_like(attn_score), attn_score)

        attn = torch.softmax(attn_score.float(), dim=-1).to(attn_score.dtype)

        # query mask
        if mask is not None:
            mask_q = mask[:, None, :, None].float()
            attn = attn * mask_q

        # 再乘一次 tree mask，确保非法边为0
        if tree_mask is not None and self.use_tree_mask:
            attn = attn * tree_mask[:, None, :, :].float()

        attn = self.dropout(attn)

        if self.value_emb_flag:
            value_hop_emb = self.topology_value_emb.weight.view(1, num_hop_types, self.nheads, d_k).transpose(1, 2)
            value_edge_emb = self.edge_value_emb.weight.view(1, num_edge_types, self.nheads, d_k).transpose(1, 2)

            value_hop_att = torch.zeros((batch_size, self.nheads, seq_len, num_hop_types), device=q.device, dtype=attn.dtype)
            value_hop_att = torch.scatter_add(
                value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1), attn
            )

            value_edge_att = torch.zeros((batch_size, self.nheads, seq_len, num_edge_types), device=q.device, dtype=attn.dtype)
            value_edge_att = torch.scatter_add(
                value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1), attn
            )

        x = torch.matmul(attn, v)

        if self.value_emb_flag:
            x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.nheads * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x
