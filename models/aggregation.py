from ctypes import sizeof
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_
from models.conv4d import Encoder4D

class SequentialMulti(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

class DWConv(nn.Module):
    def __init__(self, dim, size) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.size = size

    def forward(self, x):
        x = rearrange(x, 'B (H W) C -> B C H W', H=self.size[0], W=self.size[1])
        x = self.dwconv(x)
        x = rearrange(x, 'B C H W -> B (H W) C')
        return x

def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow
def interpolate4d(x, shape):
    B, _, H_s, W_s, _, _ = x.shape
    x = rearrange(x, 'B C H_s W_s H_t W_t -> B (C H_s W_s) H_t W_t')
    x = F.interpolate(x, size=shape[-2:], mode='bilinear', align_corners=True)
    x = rearrange(x, 'B (C H_s W_s) H_t W_t -> B (C H_t W_t) H_s W_s', H_s=H_s, W_s=W_s)
    x = F.interpolate(x, size=shape[:2], mode='bilinear', align_corners=True)
    x = rearrange(x, 'B (C H_t W_t) H_s W_s -> B C H_s W_s H_t W_t', H_t=shape[-2], W_t=shape[-1])
    return x

def interpolate2d_token(x, shape):
    B, L, C = x.shape
    x = rearrange(x, 'B (H W) C -> B C H W', H=int(L**0.5))
    x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
    x = rearrange(x, 'B C H W -> B (H W) C')
    return x


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


def correlation(src_feat, trg_feat, eps=1e-5):
    src_feat = src_feat / (src_feat.norm(dim=1, p=2, keepdim=True) + eps)
    trg_feat = trg_feat / (trg_feat.norm(dim=1, p=2, keepdim=True) + eps)

    return torch.einsum('bchw, bcxy -> bhwxy', src_feat, trg_feat)


def correlation_token(src_feat, trg_feat, feat_size, eps=1e-5):
    src_feat = rearrange(src_feat, 'B (H W) C -> B C H W', H=feat_size[0], W=feat_size[1])
    trg_feat = rearrange(trg_feat, 'B (H W) C -> B C H W', H=feat_size[0], W=feat_size[1])
    return correlation(src_feat, trg_feat)[:, None]



class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()
    
def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def soft_argmax(corr, beta=0.02):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''

    b,_,h,w = corr.size()
   
    corr = softmax_with_temperature(corr, beta=0.02, d=1) 
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = torch.linspace(-1, 1, w).expand(b,w).to(corr.device)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = torch.linspace(-1, 1, h).expand(b,h).to(corr.device)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return grid_x, grid_y

class UFCLayer(nn.Module):
    def __init__(self,
            feat_dim,
            corr_size,
            d_model,
            nhead,
            expand_ratio=4.,
            feat_size=(16, 16),
            feat_to_corr_kwargs=None,
            attention_type='self'):
        super().__init__()
        self.d_model = d_model
        self.dim = d_model // nhead
        self.nhead = nhead
        self.attention_type = attention_type
        self.feat_size = feat_size

        # multi-head attention
        self.q_proj = nn.Linear(feat_dim + corr_size ** 2 * nhead, d_model)
        self.k_proj = nn.Linear(feat_dim + corr_size ** 2 * nhead, d_model)
        self.v_proj = nn.Linear(feat_dim, d_model)
        self.v_proj_corr = Encoder4D(
            corr_levels=(nhead, nhead),
            kernel_size=(
                (3, 3, 3, 3),
            ),
            stride=(
                (1, 1, 1, 1),
            ),
            padding=(
                (1, 1, 1, 1),
            ),
            group=(1,),
        )
     
        self.attention = LinearAttention()
 
        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * expand_ratio)),
            DWConv(int(d_model * expand_ratio), feat_size),
            nn.GELU(),
            nn.Linear(int(d_model * expand_ratio), d_model),
        )
        self.mlp_corr = Encoder4D(
            corr_levels=(nhead, nhead * 4, nhead),
            kernel_size=(
                (3, 3, 3, 3),
                (3, 3, 3, 3),
            ),
            stride=(
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ),
            padding=(
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ),
            group=(1, 1),
        )


        self.mlp_cross = nn.Sequential(
            nn.Linear(d_model, int(d_model * expand_ratio)),
            DWConv(int(d_model * expand_ratio), feat_size),
            nn.GELU(),
            nn.Linear(int(d_model * expand_ratio), d_model),
        )

        # Correlation Refinement
        self.mlp_refine_corr = Encoder4D(
            corr_levels=(nhead, nhead * 4, nhead),
            kernel_size=(
                (3, 3, 3, 3),
                (3, 3, 3, 3),
            ),
            stride=(
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ),
            padding=(
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ),
            group=(1, 1),
        )
        self.mlp_refine_corr2 = Encoder4D(
            corr_levels=(nhead, nhead * 4, nhead),
            kernel_size=(
                (3, 3, 3, 3),
                (3, 3, 3, 3),
            ),
            stride=(
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ),
            padding=(
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ),
            group=(1, 1),
        )

        self.feat_to_corr1 = Encoder4D(
            **feat_to_corr_kwargs
        )
        self.feat_to_corr2 = Encoder4D(
            **feat_to_corr_kwargs
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Modules for cross attention
        self.v_cross = nn.Linear(d_model, d_model)
        self.norm_cross1 = nn.LayerNorm(d_model)
        self.norm_cross2 = nn.LayerNorm(d_model)

        # Position embedding

        self.pos_embed = nn.Parameter(torch.zeros(1, self.feat_size[0] ** 2, 1, self.dim))
        trunc_normal_(self.pos_embed, std=.02)
    def forward_attention(self, corr, feat):
        B, _, H_s, W_s, H_t, W_t = corr.shape

        feat_r = feat.clone()
        feat = self.norm1(feat)
        '''
        corr_cat = rearrange(corr, 'B H H_s W_s H_t W_t -> B (H H_t W_t) H_s W_s')
        corr_cat = F.interpolate(corr_cat, size=self.feat_size, mode='bilinear', align_corners=True)
        cf = torch.cat((rearrange(corr_cat, 'B C H_s W_s -> B (H_s W_s) C'), feat), dim=-1)
        freqs = self.pos_embed(torch.linspace(-1,1, steps = cf.shape[1],  device = feat.device), cache_key = cf.shape[1])
        freqs = rearrange(freqs[:cf.shape[1]], 'n d -> () () n d')
    
        query = apply_rotary_emb(freqs, self.q_proj(cf).view(B, -1, self.nhead, self.dim).transpose(1,2)).transpose(1,2)  # B seqlen head dimension of head
        key =  apply_rotary_emb(freqs, self.k_proj(cf).view(B, -1, self.nhead, self.dim).transpose(1,2)).transpose(1,2) 
        '''
        corr_cat = rearrange(corr, 'B H H_s W_s H_t W_t -> B (H H_t W_t) H_s W_s')
        corr_cat = F.interpolate(corr_cat, size=self.feat_size, mode='bilinear', align_corners=True)
        cf = torch.cat((rearrange(corr_cat, 'B C H_s W_s -> B (H_s W_s) C'), feat), dim=-1)

        query = self.q_proj(cf).view(B, -1, self.nhead, self.dim) + self.pos_embed
        key = self.k_proj(cf).view(B, -1, self.nhead, self.dim) + self.pos_embed
        value_feat = self.v_proj(feat).view(B, -1, self.nhead, self.dim)
        value_corr = self.v_proj_corr(corr)
        value_corr = rearrange(value_corr, 'B H H_s W_s H_t W_t -> B (H H_t W_t) H_s W_s')
        value_corr = F.interpolate(value_corr, size=self.feat_size, mode='bilinear', align_corners=True)
        value_corr = rearrange(value_corr, 'B (H H_t W_t) H_s W_s -> B (H_s W_s) H (H_t W_t)', H_t=H_t, W_t=W_t)

        msg_feat = self.attention(query, key, value_feat).view(B, -1, self.nhead*self.dim)
        msg_corr = self.attention(query, key, value_corr)
        msg_corr = rearrange(msg_corr, 'B (H_s W_s) H (H_t W_t) -> B (H H_t W_t) H_s W_s', H_s=self.feat_size[0], H_t=H_t)
        msg_corr = F.interpolate(msg_corr, size=(H_s, W_s), mode='bilinear', align_corners=True)
        msg_corr = rearrange(msg_corr, 'B (H H_t W_t) H_s W_s -> B H H_s W_s H_t W_t', H_t=H_t, W_t=W_t)

        # Residual
        msg_feat = feat_r + msg_feat
        msg_corr = corr + msg_corr


        msg_feat = msg_feat + self.mlp(self.norm2(msg_feat)) # MLP
        msg_corr = msg_corr + self.mlp_corr(msg_corr) # Correlation MLP

        return msg_corr, msg_feat

    def forward_cross(self, corr, src_feat, trg_feat):
        B, _, H_s, W_s, H_t, W_t = corr.shape
        corr = rearrange(corr, 'B Head H_s W_s H_t W_t -> B Head (H_s W_s) (H_t W_t)')

        src_feat_r = rearrange(src_feat, 'B (H W) C -> B C H W', H=self.feat_size[0])
        src_feat_r = reduce(src_feat_r, 'B C (H P1) (W P2) -> B (H W) C', P1=self.feat_size[0]//H_s, P2=self.feat_size[0]//W_s, reduction='mean')
        trg_feat_r = rearrange(trg_feat, 'B (H W) C -> B C H W', H=self.feat_size[0])
        trg_feat_r = reduce(trg_feat_r, 'B C (H P1) (W P2) -> B (H W) C', P1=self.feat_size[0]//H_t, P2=self.feat_size[0]//W_t, reduction='mean')

        trg = self.v_cross(self.norm_cross1(trg_feat_r)).view(B, -1, self.nhead, self.dim)
        src = self.v_cross(self.norm_cross1(src_feat_r)).view(B, -1, self.nhead, self.dim)

        src_attn = torch.einsum('bhst, bthc -> bshc', corr.softmax(-1), trg).reshape(B, -1, self.nhead*self.dim)
        trg_attn = torch.einsum('bhst, bshc -> bthc', corr.softmax(-2), src).reshape(B, -1, self.nhead*self.dim)

        src_attn = rearrange(src_attn, 'B (H W) C -> B C H W', H=H_s)
        src_attn = repeat(src_attn, 'B C H W -> B C (H P1) (W P2)', P1=self.feat_size[0]//H_s, P2=self.feat_size[0]//W_s)
        src_attn = rearrange(src_attn, 'B C H W -> B (H W) C')
        trg_attn = rearrange(trg_attn, 'B (H W) C -> B C H W', H=H_t)
        trg_attn = repeat(trg_attn, 'B C H W -> B C (H P1) (W P2)', P1=self.feat_size[0]//H_t, P2=self.feat_size[0]//W_t)
        trg_attn = rearrange(trg_attn, 'B C H W -> B (H W) C')

        src_feat = src_feat + src_attn
        trg_feat = trg_feat + trg_attn

        src_feat = src_feat + self.mlp_cross(self.norm_cross2(src_feat))
        trg_feat = trg_feat + self.mlp_cross(self.norm_cross2(trg_feat))
        
        return src_feat, trg_feat
    
    def forward(self, corr, src_feat, trg_feat, refine_last_corr=True):
        corr_src, src_feat_refined = self.forward_attention(corr, src_feat)
        corr_trg, trg_feat_refined = self.forward_attention(rearrange(corr, 'B H H_s W_s H_t W_t -> B H H_t W_t H_s W_s'), trg_feat)

        corr_r = corr_src + rearrange(corr_trg, 'B H H_t W_t H_s W_s -> B H H_s W_s H_t W_t')
        corr_r = corr_r + self.feat_to_corr1(correlation_token(src_feat_refined, trg_feat_refined, self.feat_size))
        corr_r = corr_r + self.mlp_refine_corr(corr_r)

        src_feat_refined, trg_feat_refined = self.forward_cross(corr_r, src_feat_refined, trg_feat_refined)

        if refine_last_corr:
            corr_r = corr_r + self.feat_to_corr2(correlation_token(src_feat_refined, trg_feat_refined, self.feat_size))
            corr_r = corr_r + self.mlp_refine_corr2(corr_r)

        return corr_r, src_feat_refined, trg_feat_refined
    
class UFC(nn.Module):
    def __init__(self, nhead=8, feat_dim=[256, 256, 256]) -> None:
        super().__init__()
        

        self.layer_nums = [2, 2, 1]
        self.layer_nums_cum = np.cumsum(self.layer_nums)
        self.feat_dim = feat_dim

        self.layers = nn.ModuleList([
            SequentialMulti(*[
                UFCLayer(
                    feat_dim=feat_dim[0],
                    corr_size=16,
                    d_model=feat_dim[0],
                    nhead=nhead,
                    feat_size=(16, 16),
                    feat_to_corr_kwargs={
                        'corr_levels': (1, nhead),
                        'kernel_size': (
                            (3, 3, 3, 3),
                        ),
                        'stride': (
                            (1, 1, 1, 1),
                        ),
                        'padding': (
                            (1, 1, 1, 1),
                        ),
                        'group': (1,)
                    }
                ) for _ in range(self.layer_nums[0])
            ]), 
            SequentialMulti(*[
                UFCLayer(
                    feat_dim=feat_dim[1],
                    corr_size=16,
                    d_model=feat_dim[1],
                    nhead=nhead,
                    feat_size=(32, 32),
                    feat_to_corr_kwargs={
                        'corr_levels': (1, nhead),
                        'kernel_size': (
                            (3, 3, 3, 3),
                        ),
                        'stride': (
                            (2, 2, 2, 2),
                        ),
                        'padding': (
                            (1, 1, 1, 1),
                        ),
                        'group': (1,)
                    }
                ) for _ in range(self.layer_nums[1])
            ]), 
            SequentialMulti(*[
                UFCLayer(
                    feat_dim=feat_dim[2],
                    corr_size=16,
                    d_model=feat_dim[2],
                    nhead=nhead,
                    feat_size=(64, 64),
                    feat_to_corr_kwargs={
                        'corr_levels': (1, nhead),
                        'kernel_size': (
                            (5, 5, 5, 5),
                        ),
                        'stride': (
                            (4, 4, 4, 4),
                        ),
                        'padding': (
                            (2, 2, 2, 2),
                        ),
                        'group': (1,)
                    }
                ) for _ in range(self.layer_nums[2])
            ])
        ])

        self.embedding = nn.ModuleList([
            Encoder4D(
                corr_levels=(1, nhead),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (1, 1, 1, 1),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(1,)
            ),
            Encoder4D(
                corr_levels=(1, nhead),
                kernel_size=(
                    (3, 3, 3, 3),
                ),
                stride=(
                    (2, 2, 2, 2),
                ),
                padding=(
                    (1, 1, 1, 1),
                ),
                group=(1,)
            ),
            Encoder4D(
                corr_levels=(1, nhead),
                kernel_size=(
                    (5, 5, 5, 5),
                ),
                stride=(
                    (4, 4, 4, 4),
                ),
                padding=(
                    (2, 2, 2, 2),
                ),
                group=(1,)
            ),
        ])


        self.proj_feat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
            )
        ])

        self.pool = nn.AvgPool2d(2)

       
        self.sigmoid = nn.Sigmoid()
    
    def forward_backbone(self, img):
        if self.freeze:
            with torch.no_grad():
                self.backbone.eval()
                feat_list = self.backbone(img)
        else:
            feat_list = self.backbone(img)
        
        return feat_list

    def forward(self, feat, nview):
      
        
        B, _, _, _ = feat[0].shape
        # [0] ctxt1 [1] ctxt2 
        # [0] ctxt2 [1] query
       
        
        src_feats = [self.proj_feat[i](rearrange(feat[i].view(B//nview, nview, -1, feat[i].shape[-1], feat[i].shape[-1])[:,0], 'B C H W -> B (H W) C')) for i in range(len(feat))]
        trg_feats = [self.proj_feat[i](rearrange(feat[i].view(B//nview, nview, -1, feat[i].shape[-1], feat[i].shape[-1])[:,1], 'B C H W -> B (H W) C')) for i in range(len(feat))]

        correlations = []
        feat_list = []
      
    
        corr_4 = correlation(rearrange(src_feats[0], 'B (H W) C -> B C H W', H = feat[0].shape[-1]), rearrange(trg_feats[0], 'B (H W) C -> B C H W' ,H = feat[0].shape[-1]))[:, None]
        corr_4 = self.embedding[0](corr_4)
        src_feat_4, trg_feat_4 = src_feats[0], trg_feats[0]
        corr_4, src_feat_4, trg_feat_4 = self.layers[0](corr_4, src_feat_4, trg_feat_4)
        feat_list.append(rearrange(torch.stack((src_feat_4, trg_feat_4), dim=1).flatten(0,1), 'B (H W) C -> B C H W', H = feat[0].shape[-1] ))
   
        correlations.append(correlation_token(src_feat_4, trg_feat_4, (16, 16)))
       
        corr_3 = correlation(rearrange(src_feats[1], 'B (H W) C -> B C H W' ,H = feat[1].shape[-1]), rearrange(trg_feats[1], 'B (H W) C -> B C H W' ,H = feat[1].shape[-1]))[:, None]
        corr_3 = corr_4 + self.embedding[1](corr_3)

        src_feat_3 = interpolate2d_token((src_feat_4), (32, 32)) + src_feats[1]
        trg_feat_3 = interpolate2d_token((trg_feat_4), (32, 32)) + trg_feats[1]
        corr_3, src_feat_3, trg_feat_3 = self.layers[1](corr_3, src_feat_3, trg_feat_3)
        feat_list.append(rearrange(torch.stack((src_feat_3, trg_feat_3), dim=1).flatten(0,1), 'B (H W) C -> B C H W', H = feat[1].shape[-1] ))
        correlations.append(correlation_token(src_feat_3, trg_feat_3, (32, 32)))
       
        
        corr_2 = correlation(rearrange(src_feats[2], 'B (H W) C -> B C H W', H = feat[2].shape[-1]), rearrange(trg_feats[2], 'B (H W) C -> B C H W', H = feat[2].shape[-1]))[:, None]
        corr_2 = corr_3 + self.embedding[2](corr_2)

        src_feat_2 = interpolate2d_token((src_feat_3), (64, 64)) + src_feats[2]
        trg_feat_2 = interpolate2d_token((trg_feat_3), (64, 64)) + trg_feats[2]
        corr_2, src_feat_2, trg_feat_2 = self.layers[2](corr_2, src_feat_2, trg_feat_2)
        feat_list.append(rearrange(torch.stack((src_feat_2, trg_feat_2), dim=1).flatten(0,1), 'B (H W) C -> B C H W', H = feat[2].shape[-1] ))
        correlations.append(correlation_token(src_feat_2, trg_feat_2, (64, 64)))
        
        corr_upsampled = [interpolate4d(x, (64, 64, 64, 64)) for x in correlations]
        
        c = (sum(corr_upsampled) / len(corr_upsampled) )  
     
        grid_x_t_to_s, grid_y_t_to_s = soft_argmax(c.permute(0,1,4,5,2,3).flatten(1, 3))
        flow_t_to_s = torch.cat((grid_x_t_to_s, grid_y_t_to_s), dim=1)
        flow = unnormalise_and_convert_mapping_to_flow(flow_t_to_s) # 2 -> 1 
  
        grid_x_s_to_t, grid_y_s_to_t = soft_argmax(c.flatten(1, 3))
        flow_s_to_t = torch.cat((grid_x_s_to_t, grid_y_s_to_t), dim=1)
        flow_flip = unnormalise_and_convert_mapping_to_flow(flow_s_to_t) # 1 -> 2 
        return feat_list, (flow, flow_flip, flow_t_to_s, flow_s_to_t), c
       