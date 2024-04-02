
#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
#Libaries for calculation and processing
from einops import rearrange, repeat
from math import sqrt
from math import ceil

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1,return_attention = False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.return_attention = return_attention
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.return_attention:
            return V.contiguous(), A.contiguous()
        
        return V.contiguous()
    
    
    
    
class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1,return_attention=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        
        self.return_attention = return_attention
        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout,return_attention=self.return_attention)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        if(self.return_attention):
            out,attention_weights= self.inner_attention(
                queries,
                keys,
                values,)

            out = out.view(B, L, -1)
        
            return self.out_projection(out), attention_weights
        
        else:
            out = self.inner_attention(
            queries,
            keys,
            values,)

            out = out.view(B, L, -1)
            return self.out_projection(out)
        
        
class AttentionLayerCrossSegments(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1,return_attention=True):
        super(AttentionLayerCrossSegments, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        
        self.return_attention = return_attention
        self.inner_attention1 = FullAttention(scale=None, attention_dropout = dropout,return_attention=self.return_attention)
        self.inner_attention2 = FullAttention(scale=None, attention_dropout = dropout,return_attention=self.return_attention)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out1_projection = nn.Linear(d_values * n_heads, d_model)
        self.out2_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
       
        

    def forward(self, queries, keys, values,num_patches):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
     
        queries_projected = self.query_projection(queries).view(B, L, H, -1)
        keys_projected = self.key_projection(keys).view(B, S, H, -1)
        values_projected = self.value_projection(values).view(B, S, H, -1)
        
        """ queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)"""
        
        q = queries_projected[:, :num_patches, :, :]
        k = keys_projected[:, -num_patches:, :, :]
        v = values_projected[:, -num_patches:, :, :]
        
        L= q.shape[1]
        
        if(self.return_attention):
            
            out1,attention_weights1= self.inner_attention1(
                q,
                k,
                v,
                )
            
            
            
            out2,attention_weights2= self.inner_attention2(
                k,
                q,
                q,
                )

            out1 = out1.view(B, L, -1)
            out2 = out2.view(B, L, -1)
            concatenated = torch.cat([out1, out2], dim=1)
            
            
            return self.out1_projection(concatenated), attention_weights1, attention_weights2,
        
        
        else:
            
            out1 = self.inner_attention1(
                q,
                k,
                v,)
            
            out2,attention_weights2= self.inner_attention2(
                k,
                q,
                q,
                )
            out1 = out1.view(B, L, -1)
            out2 = out2.view(B, L, -1)
            concatenated = torch.cat([out1, out2], dim=1)
            
            return self.out_projection(out1), self.out_projection(out2)
        
        
        
        
class TwoStageAttentionLayerCrossSegments(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, dropout=0.1):
        super(TwoStageAttentionLayerCrossSegments, self).__init__()
        d_ff = d_ff or 4*d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout,return_attention=False)
        self.dim_sender = AttentionLayerCrossSegments(d_model, n_heads, dropout = dropout,return_attention=True)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout,return_attention=True)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        
        self.projection_layer = nn.Linear(2 * d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm31 = nn.LayerNorm(d_model)
        self.norm32 = nn.LayerNorm(d_model)

        self.norm41 = nn.LayerNorm(d_model)
        self.norm42 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        
            
        self.attention_details = []

        
    
    def store_attn(self,receive_weights,send_weights):
        
        self.attention_details.append({
            'att-A-B': send_weights.detach(),  # Detach tensors for storage
            'att-B-A': receive_weights.detach(),
        })
    
    
    def get_attn(self):
        return self.attention_details
        
    def reset_attention_across_channels_details(self):
        self.attention_details = []
        
    def forward(self, x):
        #input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
        batch = x.shape[0]
        seg_num = x.shape[2]
        ts_d = x.shape[1]
        d_model = x.shape[3]
        
        x_copy = x
        #Cross Time Stage: Directly apply MSA to each dimension
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        
        final_out1 = rearrange(dim_in, '(b ts_d) seg_num d_model -> b ts_d seg_num d_model', b = batch,ts_d=ts_d)
        
        
        #Cross dimension segment-segment
        segments = x_copy.reshape(batch, ts_d*seg_num , d_model)
        
        dim_enc1, att1, att2 = self.dim_sender(segments,segments,segments,seg_num)
        #dim_enc1, att1 = self.dim_sender(segments,segments,segments)
        
        self.store_attn(att1,att2)
        
        dim_enc1 = segments + self.dropout(dim_enc1)
        
        dim_enc1 = self.norm31(dim_enc1)
        #dim_enc2 = self.norm32(dim_enc2)
        
        dim_enc1 = dim_enc1 + self.dropout(self.MLP2(dim_enc1))
       # dim_enc2 = dim_enc2 + self.dropout(self.MLP2(dim_enc2))
        
        dim_enc1 = self.norm41(dim_enc1)
        #dim_enc2 = self.norm42(dim_enc2)
        
       # concatenated = torch.cat([dim_enc1, dim_enc2], dim=1)
        
        final_out2 = rearrange(dim_enc1, 'b (ts_d seg_num) d_model -> b ts_d seg_num d_model', ts_d=ts_d, seg_num=seg_num)
        
        
        gate = torch.sigmoid(final_out2) 
        final_out = gate * final_out2 + (1 - gate) * final_out1

        
        return final_out2
        
            