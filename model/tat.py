import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

from typing import Callable

import pdb
    

class Lambda(nn.Module):
    def __init__(self, function: Callable):
        super(Lambda, self).__init__()
        if callable(function):
            if isinstance(function, nn.Module):
                raise ValueError("Expected function, but found a Module instance, which is unsafe.")
            self._func = function
        else:
            raise ValueError(f"{function} is not a callable function.")

    def forward(self, *args):
        return self._func(*args)
    

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class Attention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(Attention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        attn_mask = TriangularCausalMask(B, L, device=queries.device)
        scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()
    
    
class AttentionLayer(nn.Module):
    def __init__(self, d_query, d_key, d_value, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_value, out_channels=4*d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=4*d_model, out_channels=d_value, kernel_size=1)
        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        d_hidden = d_model // n_heads

        self.attention = Attention()
        self.query_projection = nn.Linear(d_query, d_hidden * n_heads)
        self.key_projection = nn.Linear(d_key, d_hidden * n_heads)
        self.value_projection = nn.Linear(d_value, d_hidden * n_heads)
        self.out_projection = nn.Linear(d_hidden * n_heads, d_value)
        self.n_heads = n_heads
        self.linear = nn.Linear(d_value, d_model)

        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_value)
        self.norm2 = nn.LayerNorm(d_value)
        # self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, q, k, v):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        queries = self.query_projection(q).view(B, L, H, -1)
        keys = self.key_projection(k).view(B, S, H, -1)
        values = self.value_projection(v).view(B, S, H, -1)

        att_out = self.attention(queries, keys, values)
        att_out = att_out.view(B, L, -1)
        att_out = self.out_projection(att_out)

        x = v + self.dropout(att_out)
        # v = self.norm(v)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(x + y)
        y = self.linear(y)
        return y

        # y = self.dropout(self.activation(self.conv1(v.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # return y


class ConvLayer(nn.Module):
    def __init__(self, atrous_rates, num_ts_features, num_channels, out_channels=None, reverse=False, dropout=0.0, pad=True, device=None):
        super(ConvLayer, self).__init__()
        layers = [Lambda(lambda x: torch.permute(x, (0, 2, 1)))]

        for i, rate in enumerate(atrous_rates):
            in_ch = num_channels
            out_ch = num_channels
            if i == 0:
                in_ch = num_ts_features
            if out_channels is not None and i == (len(atrous_rates) - 1):
                out_ch = out_channels
            if pad:
                pad_shape = (rate, 0, 0, 0, 0, 0) if not reverse else (0, rate, 0, 0, 0, 0)
                layers.append(nn.ConstantPad3d(pad_shape, 0))
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=2, dilation=rate, device=device))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(Lambda(lambda x: torch.permute(x, (0, 2, 1))))
        self.out_channels = out_ch
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class TATEncoder(nn.Module):
    def __init__(self, t_history, num_xs, num_xt, num_xf, num_enc, n_heads, atrous_rates):
        super(TATEncoder, self).__init__()
        # length of sequence.
        self.t_dim = t_history
        self.num_xs = num_xs
        self.num_xt = num_xt
        self.num_xf = num_xf
        self.num_enc = num_enc
        self.n_heads = n_heads

        self.static = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=self.num_xs, out_features=self.num_enc),
            nn.ReLU(),
        )

        self.atrous_rates = atrous_rates
        self.conv_xt = ConvLayer(atrous_rates=self.atrous_rates, num_ts_features=self.num_xt, num_channels=self.num_enc)
        self.conv_xf = nn.Sequential(
            Lambda(lambda x: torch.permute(x, (0, 2, 1))),
            nn.ConstantPad3d((1, 0, 0, 0, 0, 0), 0),
            nn.Conv1d(self.num_xf+self.num_enc, self.num_enc, kernel_size=2, dilation=1),
            nn.ReLU(inplace=False),
            Lambda(lambda x: torch.permute(x, (0, 2, 1))),
        )

        self.align_att = AttentionLayer(d_query=self.num_enc, d_key=self.num_enc, d_value=2*self.num_enc, d_model=self.num_enc, n_heads=self.n_heads)
        self.enh_att = AttentionLayer(d_query=self.num_enc, d_key=self.num_enc, d_value=self.num_enc, d_model=self.num_enc, n_heads=self.n_heads)

        self.global_future = self._get_global_future_layer(in_features=num_enc, out_features=self.num_enc)
        self.local_future = self._get_local_future_layer()

        self.linear = nn.Linear(in_features=2*num_enc, out_features=num_enc)

        self.horizon_specific = self._get_horizon_specific(in_features=num_enc, out_features_per_horizon=self.num_enc)
        self.horizon_agnostic = self._get_horizon_agnostic(in_features=num_enc, out_features=self.num_enc)

        self.out_layer = self._get_local_mlp(in_features=3*num_enc, hidden_size=self.num_enc, out_features=self.num_enc)

    def forward(self, xs, xt, xf):
        xs_out = self.static(xs)[:,:self.t_dim,:]

        xt_out = self.conv_xt(xt)
        xf_out = self.conv_xf(torch.cat((xf, xs_out), -1))

        h_future_global = self.global_future(xf_out)
        h_future_local = self.local_future(xf_out)

        ht = torch.cat((xt_out, h_future_global.unsqueeze(1).repeat(1, self.t_dim, 1)), dim=-1)
        ht = self.linear(ht)

        query, key, value = ht, xf_out, torch.cat((ht, xf_out), -1)
        x_enc = self.align_att(query, key, value)

        query, key, value = x_enc, x_enc, x_enc
        x_enc = self.enh_att(query, key, value)

        ht_horizon_specific = self.horizon_specific(x_enc)
        ht_horizon_agnostic = self.horizon_agnostic(x_enc.mean(1))
        h = torch.cat((ht_horizon_specific, ht_horizon_agnostic, h_future_local), dim=-1)
        x_enc = self.out_layer(h)

        return x_enc, xs_out
    
    def _get_global_future_layer(self, in_features: int, out_features: int):
        return nn.Sequential(
            Lambda(lambda x: x.reshape(x.shape[0], -1)),
            nn.Linear(in_features=in_features * self.t_dim, out_features=out_features),
            nn.Tanh(),
        )
    
    def _get_local_future_layer(self):
        return nn.Tanh()

    def _get_horizon_specific(self, in_features: int, out_features_per_horizon: int):
        return nn.Sequential(
            nn.Linear(in_features, out_features_per_horizon),
            nn.ReLU(),
            Lambda(lambda x: x.reshape(x.shape[0], self.t_dim, -1)),
        )
    
    def _get_horizon_agnostic(self, in_features: int, out_features: int):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            Lambda(lambda x: x.unsqueeze(dim=1).expand(-1, self.t_dim, -1)),
        )

    def _get_local_mlp(self, in_features: int, hidden_size: int, out_features: int):
        return nn.Sequential(
            nn.Linear(in_features, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_features), nn.ReLU()
        )



class TATDecoder(nn.Module):
    def __init__(
        self, t_history, t_future, num_xf, num_enc, n_heads, global_future_dense_units, horizon_specific_units_per_horizon,
        horizon_agnostic_hidden_size, horizon_local_mlp_hidden_size, horizon_local_mlp_units_per_horizon, atrous_rates, **kwargs,
    ):
        super(TATDecoder, self).__init__()
        self.num_horizons = t_future
        self.max_horizon = t_future
        self.t_history = t_history
        self.t_future = t_future
        self.num_xf = num_xf
        self.num_enc = num_enc
        self.n_heads = n_heads
        self.atrous_rates = atrous_rates

        self.lstm = AttentionLayer(d_query=self.t_history, d_key=self.t_history, d_value=self.t_history, d_model=self.t_history, n_heads=self.n_heads)
        self.proj = nn.Linear(in_features=self.t_history, out_features=self.t_future)

        self.global_future = self._get_global_future_layer(in_features=num_enc, out_features=global_future_dense_units)
        self.local_future = self._get_local_future_layer()

        self.linear = nn.Linear(in_features=2*num_enc, out_features=num_enc)

        self.horizon_specific = self._get_horizon_specific(in_features=num_enc, out_features_per_horizon=horizon_specific_units_per_horizon)
        self.horizon_agnostic = self._get_horizon_agnostic(in_features=num_enc, out_features=horizon_agnostic_hidden_size)

        self.out_layer = self._get_local_mlp(in_features=3*num_enc, hidden_size=horizon_local_mlp_hidden_size, out_features=horizon_local_mlp_units_per_horizon)

        self.conv_xf = nn.Sequential(
            Lambda(lambda x: torch.permute(x, (0, 2, 1))),
            nn.ConstantPad3d((1, 0, 0, 0, 0, 0), 0),
            nn.Conv1d(self.num_xf+self.num_enc, self.num_enc, kernel_size=2, dilation=1),
            nn.ReLU(inplace=False),
            Lambda(lambda x: torch.permute(x, (0, 2, 1))),
        )

        self.align_att = AttentionLayer(d_query=self.num_enc, d_key=self.num_enc, d_value=2*self.num_enc, d_model=self.num_enc, n_heads=self.n_heads)
        self.enh_att = AttentionLayer(d_query=self.num_enc, d_key=self.num_enc, d_value=self.num_enc, d_model=self.num_enc, n_heads=self.n_heads)

        self.span_1 = self._get_span_1(in_features=horizon_local_mlp_units_per_horizon)


    def forward(self, xf, xs, encoded):
        xs_out = xs[:, -self.t_future:, :]

        encoded = encoded.permute(0,2,1)
        future_enc = self.lstm(encoded, encoded, encoded) # Add triangular mask
        future_enc = self.proj(future_enc).permute(0,2,1)

        xf_out = self.conv_xf(torch.cat((xf, xs_out), -1))

        # pdb.set_trace()

        h_future_global = self.global_future(xf_out)
        h_future_local = self.local_future(xf_out)

        ht = torch.cat((future_enc, h_future_global.unsqueeze(1).repeat(1, self.t_future, 1)), dim=-1)
        ht = self.linear(ht)

        # pdb.set_trace()

        query, key, value = ht, xf_out, torch.cat((ht, xf_out), -1)
        future_att = self.align_att(query, key, value) 

        query, key, value = future_att, future_att, future_att
        future_att = self.enh_att(query, key, value) 

        ht_horizon_specific = self.horizon_specific(future_att)
        ht_horizon_agnostic = self.horizon_agnostic(future_att.mean(1))
        h = torch.cat((ht_horizon_specific, ht_horizon_agnostic, h_future_local), dim=-1)
        h = self.out_layer(h)

        # preds = self.span_1(torch.cat((future_enc, future_att), -1))
        preds = self.span_1(h)

        return preds
    
    def _get_global_future_layer(self, in_features: int, out_features: int):
        return nn.Sequential(
            Lambda(lambda x: x.reshape(x.shape[0], -1)),
            nn.Linear(in_features=in_features * self.t_future, out_features=out_features),
            nn.Tanh(),
        )
    
    def _get_local_future_layer(self):
        return nn.Tanh()

    def _get_horizon_specific(self, in_features: int, out_features_per_horizon: int):
        return nn.Sequential(
            nn.Linear(in_features, out_features_per_horizon),
            nn.ReLU(),
            Lambda(lambda x: x.reshape(x.shape[0], self.num_horizons, -1)),
        )
    
    def _get_horizon_agnostic(self, in_features: int, out_features: int):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            Lambda(lambda x: x.unsqueeze(dim=1).expand(-1, self.num_horizons, -1)),
        )

    def _get_local_mlp(self, in_features: int, hidden_size: int, out_features: int):
        return nn.Sequential(
            nn.Linear(in_features, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_features), nn.ReLU()
        )

    def _get_span_1(self, in_features: int):
        return nn.Sequential(
            nn.Linear(in_features, 2),
            nn.PReLU()
        )


class TAT(nn.Module):
    def __init__(self, configs):
        super(TAT, self).__init__()
        self.num_xt = configs.num_xt
        self.num_xf = configs.num_xf
        self.num_xs = configs.num_xs

        self.t_history = configs.lookback
        self.t_future = configs.horizon
        self.hidden = configs.hidden
        self.atrous_rates = configs.atrous_rates
        self.n_heads = configs.heads

        self.encoder = TATEncoder(t_history=self.t_history, num_xs=self.num_xs, num_xt=self.num_xt, num_xf=self.num_xf, \
                                 num_enc=self.hidden, n_heads=self.n_heads, atrous_rates=self.atrous_rates)
        
        self.decoder = TATDecoder(t_history=self.t_history, t_future=self.t_future, num_xf=self.num_xf, num_enc=self.hidden, n_heads=self.n_heads, \
                                global_future_dense_units=self.hidden, horizon_specific_units_per_horizon=self.hidden, \
                                horizon_agnostic_hidden_size=self.hidden, horizon_local_mlp_hidden_size=self.hidden, \
                                horizon_local_mlp_units_per_horizon=self.hidden, atrous_rates=self.atrous_rates)

    def forward(self, xt, xf, xs):
        means = xt.mean(1, keepdim=True).detach()
        xt = xt - means
        stdev = torch.sqrt(torch.var(xt, dim=1, keepdim=True, unbiased=False) + 1e-5)
        xt /= stdev

        xf_hist = xf[:, :self.t_history, :]
        xf_future = xf[:, -self.t_future:, :]

        encoded, xs_out = self.encoder(xs, xt, xf_hist)
        preds = self.decoder(xf_future, xs_out, encoded)

        preds = preds * (stdev[:, 0, :1].unsqueeze(1).repeat(1, self.t_future, 1))
        preds = preds + (means[:, 0, :1].unsqueeze(1).repeat(1, self.t_future, 1))
        return preds