import math
from functools import partial
from collections import OrderedDict
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_training import utils

class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """
    def __init__(
        self,
        backbone="resnet34",
        pretrained=False,
        num_layers=5,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = utils.get_norm_layer(norm_type)

        print("Using torchvision", backbone, "encoder")
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained
        )
        # Following 2 lines need to be uncommented for older configs
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        # self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

    def forward(self, x, cam2world, n_view):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        x = x
        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            # self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latents = latents[::-1]
            latents_large = latents

            self.latent = latents
        return self.latent




class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def get_l1_positional_encodings(B, N, intrinsics=None):

    h,w = 48,64
    if N == 24*24:
        h,w = 24,24
    elif N != 48*64:
        print('unexpected resolution for positional encoding')
        assert(False)

    positional = torch.ones([B, N, 6])

    ys = torch.linspace(-1,1,steps=h)
    xs = torch.linspace(-1,1,steps=w)
    p3 = ys.unsqueeze(0).repeat(B,w)
    p4 = xs.repeat_interleave(h).unsqueeze(0).repeat(B,1)

    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics[:,0].unbind(dim=-1)

        hpix = cy * 2
        wpix = cx * 2
        # map to between -1 and 1
        fx_normalized = (fx / wpix) * 2
        cx_normalized = (cx / wpix) * 2 - 1 
        fy_normalized = (fy / hpix) * 2
        cy_normalized = (cy / hpix) * 2 - 1
        # in fixed case, if we are mapping rectangular img with width > height,
        # then fy will be > fx and therefore p3 will be both greater than -1 and less than 1. ("y is zoomed out")
        # p4 will be -1 to 1.

        K = torch.zeros([B,3,3])
        K[:,0,0] = fx_normalized
        K[:,1,1] = fy_normalized
        K[:,0,2] = cx_normalized
        K[:,1,2] = cy_normalized
        K[:,2,2] = 1
    
        Kinv = torch.inverse(K)
        for j in range(h):
            for k in range(w):
                w1, w2, w3 = torch.split(Kinv @ torch.tensor([xs[k], ys[j], 1]), 1, dim=1)
                p3[:, int(k * w + j)] = w2.squeeze() / w3.squeeze() 
                p4[:, int(k * w + j)] = w1.squeeze() / w3.squeeze() 
        
        
    #p2 = p3 * p4
    #p1 = p4 * p4
    #p0 = p3 * p3
    positional[:,:,3:5] = torch.stack([p3,p4],dim=2)

    return positional


def get_positional_encodings(B, N, intrinsics=None):
    '''
    # we now append a positional encoding onto v
    # of dim 6 (x^2, y^2, xy, x, y, 1)
    # this way, we can model linear & non-linear
    # relations between height & width. 
    # we multiply attention by this encoding on both sides
    # the results correspond to the variables in UTU
    # from the fundamental matrix
    # so, v is of shape B, N, C + 6
    '''
    h,w = 48,64
    if N == 64*64:
        h,w = 64,64
    elif N != 48*64:
        print('unexpected resolution for positional encoding')
        assert(False)

    positional = torch.ones([B, N, 6])

    ys = torch.linspace(-1,1,steps=h)
    xs = torch.linspace(-1,1,steps=w)
    p3 = ys.unsqueeze(0).repeat(B,w)
    p4 = xs.repeat_interleave(h).unsqueeze(0).repeat(B,1)

    if intrinsics is not None:
        # make sure not changing over frames
        #assert(torch.all(intrinsics[:,0]==intrinsics[:,1]).cpu().numpy().item())

        '''
        use [x'/w', y'/w'] instead of x,y for coords. Where [x',y',w'] = K^{-1} [x,y,1]
        '''
      
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

        if cx[0] * cy[0] == 0:
            print('principal point is in upper left, not setup for this right now.')
            import pdb; pdb.set_trace()

        hpix = cy * 2
        wpix = cx * 2
        # map to between -1 and 1
        fx_normalized = (fx / wpix) * 2
        cx_normalized = (cx / wpix) * 2 - 1 
        fy_normalized = (fy / hpix) * 2
        cy_normalized = (cy / hpix) * 2 - 1
        # in fixed case, if we are mapping rectangular img with width > height,
        # then fy will be > fx and therefore p3 will be both greater than -1 and less than 1. ("y is zoomed out")
        # p4 will be -1 to 1.

        K = torch.zeros([B,3,3])
        K[:,0,0] = fx_normalized.squeeze()
        K[:,1,1] = fy_normalized.squeeze()
        K[:,0,2] = cx_normalized.squeeze()
        K[:,1,2] = cy_normalized.squeeze()
        K[:,2,2] = 1
    
        Kinv = torch.inverse(K)
        for j in range(h):
            for k in range(w):
                w1, w2, w3 = torch.split(Kinv @ torch.tensor([xs[k], ys[j], 1]), 1, dim=1)
                p3[:, int(k * w + j)] = w2.squeeze() / w3.squeeze() 
                p4[:, int(k * w + j)] = w1.squeeze() / w3.squeeze() 

    p2 = p3 * p4
    p1 = p4 * p4
    p0 = p3 * p3
    positional[:,:,:5] = torch.stack([p0,p1,p2,p3,p4],dim=2)
    
    return positional

class CrossAttention(nn.Module):
    """
    Our custom Cross-Attention Block. Have options to use dual softmax, 
    add positional encoding and use bilinear attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., 
                proj_drop=0., cross_features=False, 
                use_single_softmax=False, 
                no_pos_encoding=False, noess=False, l1_pos_encoding=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if noess:
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj_fundamental = nn.Linear(dim+int(6), dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cross_features = cross_features
        self.use_single_softmax = use_single_softmax
        self.no_pos_encoding = no_pos_encoding
        self.noess = noess
        self.l1_pos_encoding = l1_pos_encoding

    def forward(self, x1, x2, corr, camera=None, intrinsics=None):
        B, N, C = x1.shape
        
        
       
        if not self.noess:
            attn_1 = corr.squeeze(1).flatten(-2,-1).flatten(1,2) # src trg
            attn_2 = corr.squeeze(1).flatten(-2,-1).flatten(1,2).transpose(-2, -1) # trg src
            
            if self.use_single_softmax:
                attn_fundamental_1 = attn_1.softmax(dim=-1)
                attn_fundamental_2 = attn_2.softmax(dim=-1)
            else:
                attn_fundamental_1 = attn_1.softmax(dim=-1) * attn_1.softmax(dim=-2) 
                attn_fundamental_2 = attn_2.softmax(dim=-1) * attn_2.softmax(dim=-2)
            
            if self.l1_pos_encoding:
                positional = get_l1_positional_encodings(B, N, intrinsics=intrinsics).cuda() # shape B,N,6
            else:
                positional = get_positional_encodings(B, N, intrinsics=intrinsics).cuda() # shape B,N,6
            if self.no_pos_encoding:
                pass
            else:
              
                # 2 3 1024 262
                v1 = torch.cat([x1,positional],dim=2)
                v2 = torch.cat([x2,positional],dim=2)
            
            if self.cross_features:
                #2 262 786
                fundamental_1 = (v2.transpose(-2, -1) @ attn_fundamental_1) @ v1
                fundamental_2 = (v1.transpose(-2, -1) @ attn_fundamental_2) @ v2
            else:
                
                fundamental_1 = (v1.transpose(-2, -1) @ attn_fundamental_1) @ v1
                fundamental_2 = (v2.transpose(-2, -1) @ attn_fundamental_2) @ v2

            if self.no_pos_encoding:
                fundamental_1 = fundamental_1.reshape(B, int(C), int(C/self.num_heads)).transpose(-2,-1)           
                fundamental_2 = fundamental_2.reshape(B, int(C), int(C/self.num_heads)).transpose(-2,-1)
            else:
                
                fundamental_1 = fundamental_1.reshape(B, int(C+6), int((C+6))).transpose(-2,-1)           
                fundamental_2 = fundamental_2.reshape(B, int(C+6), int((C+6))).transpose(-2,-1) 
            # fundamental is C/3+6,C/3+6 (for each head)

            fundamental_2 = self.proj_fundamental(fundamental_2)
            fundamental_1 = self.proj_fundamental(fundamental_1)
            
            # we flip these: we want x1 to be (q1 @ k2) @ v2
            # impl is similar to ViLBERT
            return fundamental_2, fundamental_1
        else:
            # q2, k1, v1
            attn_1 = (q2 @ k1.transpose(-2, -1)) * self.scale
            attn_1 = attn_1.softmax(dim=-1)
            attn_1 = self.attn_drop(attn_1)

            x1 = (attn_1 @ v1).transpose(1, 2).reshape(B, N, C)

            # q1, k2, v2
            attn_2 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn_2 = attn_2.softmax(dim=-1)
            attn_2 = self.attn_drop(attn_2)

            x2 = (attn_2 @ v2).transpose(1, 2).reshape(B, N, C)
            
            x1 = self.proj(x1)
            x2 = self.proj(x2)

            x1 = self.proj_drop(x1)
            x2 = self.proj_drop(x2)

            # we flip these: we want x1 to be (q1 @ k2) @ v2
            # impl is similar to ViLBERT
            return x2, x1 


class CrossBlock(nn.Module):
    def __init__(self, dim = 256, num_heads= 3, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, cross_features=False,
                 use_single_softmax=False, 
                 no_pos_encoding=False, noess=False, l1_pos_encoding=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop,
                                cross_features=cross_features, 
                                use_single_softmax=use_single_softmax, 
                                no_pos_encoding=no_pos_encoding, noess=noess, l1_pos_encoding=l1_pos_encoding)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.noess = noess
        self.norm = norm_layer(dim)
    def forward(self, x, camera=None, corr = None,intrinsics=None):
        b_s, h_w, nf = x.shape
        x = x.reshape([-1, 2, h_w, nf])
        x1_in = x[:,0]
        x2_in = x[:,1]

        if not self.noess:
          
            fundamental1, fundamental2 = self.cross_attn(self.norm1(x1_in), self.norm1(x2_in), corr,camera, intrinsics=intrinsics)
            fundamental_inter = torch.cat([fundamental1.unsqueeze(1), fundamental2.unsqueeze(1)], dim=1)
           
            fundamental = fundamental_inter.reshape(b_s, -1, nf)
            fundamental = fundamental + self.drop_path(self.mlp(self.norm2(fundamental)))
       
            return self.norm(fundamental)
        else:
            x1, x2 = self.cross_attn(self.norm1(x1_in), self.norm1(x2_in), camera, intrinsics=intrinsics)
            x_inter = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)
            x_inter = x_inter.reshape(b_s, h_w, nf)
            x = x.reshape(b_s, h_w, nf)
            x = x + self.drop_path(x_inter)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return self.norm(x)



 