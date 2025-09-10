# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from einops import rearrange
import math


__all__ = [
'deit_base_distilled_patch16_224','deit_base_distilled_patch16_224_small'
]

class QKV_s(nn.Module):
    def __init__(self, emb, qk, v, qkv_bias=True):
        super().__init__()

        self.Q = nn.Linear(emb, qk, bias=qkv_bias)
        self.K = nn.Linear(emb, qk, bias=qkv_bias)
        self.V = nn.Linear(emb, v, bias=qkv_bias)
        self.token_mask = nn.Parameter(torch.ones(198, 1))

    def forward(self, x):
        q = self.Q(x)*self.token_mask
        k = self.K(x)*self.token_mask
        v = self.V(x)*self.token_mask
        return q,k,v

class ATT(nn.Module):
    def __init__(self, attn_drop=0., scale=None):
        super().__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = scale

    def forward(self, q, k):
        attn_r = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_r.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn

class PROJ(nn.Module):
    def __init__(self, v, dim, proj_drop=0.):
        super().__init__()
        self.v = v
        self.proj = nn.Linear(v, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, attn, v):
        x = (attn @ v).transpose(1, 2).reshape(-1, 198, self.v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, qk, v, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        
        head_dim = dim // self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = 64 ** -0.5

        self.qkv = QKV_s(dim, qk, v)
        self.att = ATT(attn_drop,self.scale)
        self.proj = PROJ(v, dim, proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x)
        qk_dim = q.shape[2]
        v_dim = v.shape[2]
        q = q.reshape(B, N, self.num_heads, qk_dim // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, qk_dim // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1, 3)

        attn = self.att(q,k)
        x = self.proj(attn,v)
        
        return x

class DistilledVisionTransformer_small(VisionTransformer):
    def __init__(self, QK, V, MLP, head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        img_size = self.patch_embed.img_size
        patch_size = self.patch_embed.patch_size
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=self.embed_dim)

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        
        for i,blk in enumerate(self.blocks):
            blk.attn = Attention(self.embed_dim, QK[i]*head[i], V[i]*head[i],head[i])
            blk.mlp.fc1 = nn.Linear(self.embed_dim, MLP[i])
            blk.mlp.fc2 = nn.Linear(MLP[i], self.embed_dim)
    
    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x,x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return x, x_dist

############################################################################################################################

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding, modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, img_size=(32,32), patch_size=(4,4), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        pos_dim = 16

        self.proj = nn.Conv2d(in_chans*patch_size[0]*patch_size[1], embed_dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = rearrange(x, 'b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = self.patch_size[0], p2 = self.patch_size[1])
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class head_mask(nn.Module): # Multihead at output
    def __init__(self, head):
        super(head_mask, self).__init__()
        
        self.head = head
        self.weight = nn.Parameter(torch.ones(head,1))

    def forward(self, input):
        x = rearrange(input, 'b t (h q) -> b t h q',h=self.head)
        x = x*self.weight
        x = rearrange(x, 'b t h q -> b t (h q)')
        return x

class MHLinear(nn.Module): # Multihead at output
    def __init__(self, in_features, out_features, head, bias=True):
        super(MHLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.head = head
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features,head))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features,head))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = rearrange(self.weight, 'q e h -> (h q) e')
        bias = rearrange(self.bias, 'q h -> (h q)')
        return nn.functional.linear(input, weight, bias)
        
class MHTLinear(nn.Module): # Multihead at input
    def __init__(self, in_features, out_features, head, bias=True):
        super(MHTLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.head = head
        self.weight = nn.Parameter(torch.Tensor(out_features,in_features,head))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = rearrange(self.weight, 'e v h -> e (h v)')
        return nn.functional.linear(input, weight, self.bias)


class QKV(nn.Module):
    def __init__(self, emb, qk, v, head, qkv_bias=True):
        super().__init__()
        
        self.head_mask = head_mask(head)
        self.Q = MHLinear(emb, qk, head, bias=qkv_bias)
        self.K = MHLinear(emb, qk, head, bias=qkv_bias)
        self.V = MHLinear(emb, v, head, bias=qkv_bias)
        self.token_mask = nn.Parameter(torch.ones(198, 1))

    def forward(self, x):
        q = self.head_mask(self.Q(x)*self.token_mask)
        k = self.head_mask(self.K(x)*self.token_mask)
        v = self.head_mask(self.V(x)*self.token_mask)
        
        x = torch.cat((q,k,v),-1)
        return x 


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        img_size = self.patch_embed.img_size
        patch_size = self.patch_embed.patch_size
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=self.embed_dim)
        
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        # self.pos = (kernel_size==0)
        self.pos = 1
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        
        for blk in self.blocks:
            blk.attn.qkv = QKV(self.embed_dim, 64, 64, 12) 
            blk.attn.proj = MHTLinear(64, self.embed_dim, 12)
        
    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        
        if self.pos:
            x = x + self.pos_embed
        else:
            x[:, 0:1] = x[:, 0:1] + self.pos_embed[:, 0:1]
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x) #[0]

        x = self.norm(x)
        
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x,x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return x, x_dist


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    # del kwargs['pretrained_cfg']
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_model = checkpoint["model"]
        embedding = checkpoint_model['patch_embed.proj.weight']
        new_embedding = rearrange(embedding, 'b c p1 p2 -> b (p1 p2 c) 1 1')
        checkpoint_model['patch_embed.proj.weight'] = new_embedding
        for i in range(12):
            var_name = 'blocks.'+str(i)+'.attn.qkv'
            var_name2 = 'blocks.'+str(i)+'.attn.proj'
            qkv_w = checkpoint_model.pop(var_name+'.weight')
            qkv_b = checkpoint_model.pop(var_name+'.bias')
            proj_w = checkpoint_model.pop(var_name2+'.weight')
            N,C = qkv_w.shape
            qkv_w = qkv_w.reshape(3,-1,C)
            qkv_b = qkv_b.reshape(3,-1)
            q_w, k_w, v_w = qkv_w[0], qkv_w[1], qkv_w[2]
            q_b, k_b, v_b = qkv_b[0], qkv_b[1], qkv_b[2]
            checkpoint_model[var_name+'.Q.weight'] = rearrange(q_w, '(h q) e -> q e h',h=12)
            checkpoint_model[var_name+'.K.weight'] = rearrange(k_w, '(h q) e -> q e h',h=12)
            checkpoint_model[var_name+'.V.weight'] = rearrange(v_w, '(h q) e -> q e h',h=12)
            checkpoint_model[var_name+'.Q.bias'] = rearrange(q_b, '(h q) -> q h',h=12)
            checkpoint_model[var_name+'.K.bias'] = rearrange(k_b, '(h q) -> q h',h=12)
            checkpoint_model[var_name+'.V.bias'] = rearrange(v_b, '(h q) -> q h',h=12)
            checkpoint_model[var_name2+'.weight'] = rearrange(proj_w, 'e (h v) -> e v h',h=12)
        
        model.load_state_dict(checkpoint_model, strict=False)
    return model

@register_model
def deit_base_distilled_patch16_224_small(pretrained=False, **kwargs):
    EMB = kwargs.pop('EMB', None)
    QK = kwargs.pop('QK', None)
    V = kwargs.pop('V', None)
    MLP = kwargs.pop('MLP', None)
    head = kwargs.pop('head', None)
    
    model = DistilledVisionTransformer_small(
        QK=QK, V=V, MLP=MLP, head=head, patch_size=16, embed_dim=EMB, depth=12, num_heads=1, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model



