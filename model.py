import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
import torch_geometric.nn
import torch_scatter
from typing import List

# -----------------------------------------------------------------------------
# Auxiliary Components (reuse EGNN backbone)
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=None, num_layers=2, act="silu"):
        super().__init__()
        self.hidden_size = hidden_size or in_dim
        if act == "silu":
            self.activation = nn.SiLU()
        elif act == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, self.hidden_size))
            layers.append(self.activation)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers.append(self.activation)
            layers.append(nn.Linear(self.hidden_size, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class AtomLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_channels, act="silu", aggr="mean"):
        super().__init__()
        self.aggr = aggr
        self.m_msg = hidden_channels
        self.sqrt_mmsg = self.m_msg ** 0.5
        self.m_hidden = hidden_channels

        self.LnQ = nn.Linear(in_channels, self.m_msg)
        self.LnK = nn.Linear(in_channels, in_channels)
        self.LnV = nn.Linear(in_channels, in_channels)
        self.LnE = nn.Linear(edge_channels, edge_channels)

        activation = nn.SiLU() if act == "silu" else nn.ReLU()

        self.sigma = nn.Sequential(
            nn.Linear(in_channels + edge_channels, self.m_hidden),
            activation,
            nn.Linear(self.m_hidden, self.m_msg),
            activation
        )
        self.phi_msg = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_channels, self.m_hidden),
            activation,
            nn.Linear(self.m_hidden, self.m_msg),
            activation
        )
        self.phi_out = nn.Sequential(
            nn.Linear(self.m_msg + in_channels, self.m_hidden),
            activation,
            nn.Linear(self.m_hidden, out_channels)
        )
        self.alpha = nn.Sequential(
            nn.Linear(6 + 2 * in_channels, self.m_hidden),
            activation,
            nn.Linear(self.m_hidden, 3)
        )
        self.beta = nn.Sequential(
            nn.Linear(6 + 2 * in_channels, self.m_hidden),
            activation,
            nn.Linear(self.m_hidden, 3)
        )
        self.gate = nn.Sequential(
            nn.Linear(6 + 2 * in_channels, self.m_hidden),
            activation,
            nn.Linear(self.m_hidden, 3)
        )

    def forward(self, h, pos, edge_index, edge_attr, edge_vec, pos_frac):
        row, col = edge_index
        q = self.LnQ(h[row])
        k = self.sigma(torch.cat((self.LnE(edge_attr), self.LnK(h[col])), dim=1))
        w = torch.sigmoid(q * k / self.sqrt_mmsg)

        msg = self.phi_msg(torch.cat([self.LnE(edge_attr), self.LnV(h[row]), self.LnV(h[col])], dim=1))
        m_ij = w * msg
        if self.aggr == "mean":
            agg_m = torch_scatter.scatter_mean(m_ij, col, dim=0, dim_size=h.size(0))
        else:
            agg_m = torch_scatter.scatter_sum(m_ij, col, dim=0, dim_size=h.size(0))
        h_out = self.phi_out(torch.cat([agg_m, h], dim=1)) + h

        tempv = torch.cat((
            (edge_vec * pos_frac[row]).sum(dim=1, keepdim=True),
            (edge_vec * pos_frac[col]).sum(dim=1, keepdim=True),
            (pos_frac[row] * pos_frac[col]).sum(dim=1, keepdim=True),
            torch.norm(edge_vec, dim=1, keepdim=True),
            torch.norm(pos_frac[row], dim=1, keepdim=True),
            torch.norm(pos_frac[col], dim=1, keepdim=True),
            h[col], h[row]),
            dim=1)
        a = self.alpha(tempv)
        b = torch.sigmoid(self.beta(tempv))
        g = torch.tanh(self.gate(tempv))
        modulated_edge_vec = edge_vec * g
        weighted_edge_vec = a * modulated_edge_vec
        weighted_pos_col = b * pos[col]
        delta_pos = torch_scatter.scatter_mean(weighted_edge_vec + weighted_pos_col, col, dim=0, dim_size=h.size(0))
        pos_out = pos + delta_pos
        return h_out, pos_out

class EGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, edge_channels):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList([AtomLayer(hidden_channels, hidden_channels, hidden_channels, edge_channels) for _ in range(num_layers)])
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.pool = torch_geometric.nn.global_mean_pool

    def forward(self, data: Data):
        x, pos, edge_index, edge_attr, edge_vec, pos_frac = data.x, data.pos, data.edge_index, data.edge_attr, data.edge_vec, data.pos_frac
        batch = getattr(data, 'batch', torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device))
        h = self.lin_in(x)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr, edge_vec, pos_frac)
        h = self.lin_out(h)
        pooled_h = self.pool(h, batch)
        return h, pooled_h

# -----------------------------------------------------------------------------
# Temporal Transformer-based Predictor
# -----------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, timesteps: torch.Tensor):
        return self.pe[timesteps]

class PerAtomDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_out   = nn.Linear(hidden_dim, hidden_dim)
        self.ffn        = MLP(hidden_dim, hidden_dim, hidden_size=hidden_dim)
        self.pos_head   = nn.Linear(hidden_dim, 3)
        self.norm1      = nn.LayerNorm(hidden_dim)
        self.norm2      = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h0: torch.Tensor, global_z: torch.Tensor):
        B, N, C = atom_h0.shape
        context = global_z.unsqueeze(1).expand(B, N, C)
        Q = self.query_proj(atom_h0)
        K = self.key_proj(context)
        V = self.value_proj(context)
        attn_scores = (Q * K).sum(dim=-1) / (C ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
        attn_out = self.attn_out(V * attn_weights)
        x = self.norm1(atom_h0 + attn_out)
        x = self.norm2(x + self.ffn(x))
        delta_pos = self.pos_head(x)
        return delta_pos

class TemporalEGNN(nn.Module):
    def __init__(self, hidden_channels, edge_channels, n_atom_features, device, num_gnn_layers=5, num_transformer_layers=4, num_heads=8):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_channels
        self.egnn = EGNN(n_atom_features, hidden_channels, hidden_channels, num_gnn_layers, edge_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels * 4,
            batch_first=True,
            activation="gelu",
            norm_first=False
        )
        self.time_embed = SinusoidalTimeEmbedding(hidden_channels)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_channels))
        nn.init.normal_(self.cls_token, std=0.02)
        self.atom_decoder = PerAtomDecoder(hidden_channels)

    def forward(self, batch_sequences: List[List[Data]]):
        B = len(batch_sequences)
        if B == 0:
            return torch.empty(0, 3, device=self.device)

        pooled_per_seq, seq_lens, h0_atoms, n_nodes_0, pos0_list = [], [], [], [], []

        for s_idx, seq in enumerate(batch_sequences):
            cur_pooled = []
            for t, data_obj in enumerate(seq[:-1]):
                data_obj = data_obj.to(self.device)
                node_h, pooled_h = self.egnn(data_obj)
                cur_pooled.append(pooled_h.squeeze(0))
                if t == 0:
                    h0_atoms.append(node_h)
                    n_nodes_0.append(data_obj.num_nodes)
                    pos0_list.append(data_obj.pos)
            pooled_per_seq.append(torch.stack(cur_pooled, dim=0))
            seq_lens.append(len(cur_pooled))

        padded = torch.nn.utils.rnn.pad_sequence(pooled_per_seq, batch_first=True)
        max_T = padded.size(1)
        time_indices = torch.arange(max_T, device=self.device).unsqueeze(0).expand(B, -1)
        padded = padded + self.time_embed(time_indices)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, padded], dim=1)
        key_padding_mask = torch.zeros(B, max_T + 1, dtype=torch.bool, device=self.device)
        for i, L in enumerate(seq_lens):
            key_padding_mask[i, 1+L:] = True
        z = self.transformer(x, src_key_padding_mask=key_padding_mask)
        z_cls = z[:, 0]

        h0_cat = torch.cat(h0_atoms, dim=0)
        batch_index = torch.cat([torch.full((n,), i, dtype=torch.long, device=self.device) for i, n in enumerate(n_nodes_0)])
        h0_padded, h0_batch = to_dense_batch(h0_cat, batch_index)
        delta_pos = self.atom_decoder(h0_padded, z_cls)
        pos0_cat = torch.cat(pos0_list, dim=0)
        pos0_padded, _ = to_dense_batch(pos0_cat, batch_index)
        pred_pos = pos0_padded + delta_pos
        pred_flat = pred_pos[h0_batch]
        return pred_flat
