"""
Implementations of Video Transformers in PyTorch

A PyTorch implementation of space-time transformer as described in
'Frozen in Time: A Joint Image and Video Encoder for End-to-End Retrieval' - https://arxiv.org/abs/2104.00650

A PyTorch implementation of timesformer as described in
'Is Space-Time Attention All You Need for Video Understanding?' - https://arxiv.org/abs/2102.05095

Acknowledgments:
- This code builds on Ross Wightman's vision_transformer code in pytorch-image-models:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

- It is also inspired by lucidrains timesformer implementation:
https://github.com/lucidrains/TimeSformer-pytorch

Hacked together by Max Bain
"""

from collections import OrderedDict
from functools import partial

import torch
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn
from lavis.models.eva_vit import create_eva_vit_g

# The names of CLIP layers (for freezing)
CLIP_INIT_LAYERS = [
    "patch_embed",
    "norm_pre",
    "norm1",
    "attn",
    "drop_path",
    "norm2",
    "mlp",
    "norm3", # CLIP doesn't have norm3, but we add it so that the ViT output stays unchanged during training
    "norm",
    "cls_token",
    "pos_embed",
]


class PatchDropout(torch.nn.Module):
    """ 
    Implementation modified from: https://github.com/yueliukth/PatchDropout/blob/main/scripts/patchdropout.py
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    Adds capability to sample tokens from tubelets (i.e. identical spatial location in consecutive frames)
    in addition to the regular sampling from single frames.
    """
    def __init__(self, p=0.0, sampling="tubelet_uniform", token_shuffling=False, tokens_per_frame=196, num_frames=4):
        super().__init__()
        assert 0 <= p < 1, "The dropout rate p must be in [0,1)"
        self.tokens_per_frame = tokens_per_frame
        self.keep_rate = 1 - p
        self.sampling = sampling
        self.token_shuffling = token_shuffling
        self.num_frames = num_frames
        self.n_keep = int(self.tokens_per_frame * (1 - p)) # number of frames to keep per patch (if tubelet sampling is used)

    def forward(self, x, force_drop=False):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        if not self.training and not force_drop: return x        
        if self.keep_rate == 1: return x

        # batch, length, dim
        N, L, D = x.shape
        
        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x
    
    def get_mask(self, x):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        elif self.sampling == "tubelet_uniform":
            return self.tubelet_uniform_mask(x)
        else:
            raise NotImplementedError(f"PatchDropout does not support {self.sampling} sampling")
            return None

    def uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        
        keep = self.n_keep
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask
    
    def tubelet_uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        if L == self.tokens_per_frame + 1: # Image input
            return self.uniform_mask(x)
        _L = self.tokens_per_frame # patch length (without CLS)
        keep = self.n_keep
        #import pdb; pdb.set_trace()
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) 
        patch_mask = patch_mask[:, :keep]
        # Repeat the same mask for all frames for mask tubelets
        repeated_patch_mask = patch_mask.repeat(1, self.num_frames)
        values_to_add = self.tokens_per_frame * torch.arange(0, self.num_frames).repeat_interleave(keep).to(x.device)

        patch_mask = repeated_patch_mask + values_to_add 
        patch_mask = patch_mask + 1 # add 1 to account for CLS token (assumes it is leading token)
        
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        else:
            raise NotImplementedError("Token shuffling is not implemented for tubelet_uniform_mask")
        return patch_mask

def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Mlp(nn.Module):
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


class VideoPatchEmbed(nn.Module):
    """ Video to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=8):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, F, C, H, W = x.shape
        assert F <= self.num_frames
        x = x.view(-1, C, H, W)
        x = self.proj(x)
        return x


class VarAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        # project x to q, k, v values
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale

        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        ## to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class GatedTimeVarAttention(VarAttention):

    def __init__(self, *args, patches_per_frame=196, patches_per_frame_after_dropout=196, **kwargs):
        super().__init__(*args, **kwargs)
        self.patches_per_frame=patches_per_frame
        self.patches_per_frame_after_dropout=patches_per_frame_after_dropout

    """
    Modifies the forward pass of the regular attention block so that the tokens pass through, unmodified when only a single frame is input
    """

    def forward(self, x, einops_from, einops_to, **einops_dims):
        #import pdb; pdb.set_trace()
        num_input_patches = x.size(1) - 1
        if num_input_patches == self.patches_per_frame and not self.training:
            return x
        if num_input_patches == self.patches_per_frame_after_dropout and self.training:
            return x
        h = self.num_heads
        # project x to q, k, v values
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale
        
        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        ## to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class SpaceTimeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_init='zeros',
                 attention_style='frozen-in-time', patches_per_frame=196, patches_per_frame_after_dropout=196):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        if attention_style == 'freeze-first-frame':
            self.timeattn = GatedTimeVarAttention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            initialize=time_init, patches_per_frame=patches_per_frame, patches_per_frame_after_dropout=patches_per_frame_after_dropout)
        else:
            self.timeattn = VarAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                initialize=time_init)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

        self.attention_style = attention_style

    def forward(self, x, einops_from_space, einops_to_space, einops_from_time, einops_to_time,
                time_n, space_f):
        a = self.norm3(x)
        time_output = self.timeattn(a, einops_from_time, einops_to_time, n=time_n)
        time_residual = x + time_output
        space_output = self.attn(self.norm1(time_residual), einops_from_space,
                                 einops_to_space, f=space_f)
        if self.attention_style == 'frozen-in-time' or self.attention_style == 'freeze-first-frame':
            space_residual = x + self.drop_path(space_output)
        else:
            raise NotImplementedError

        x = space_residual + self.drop_path(self.mlp(self.norm2(space_residual)))

        return x


class SpaceTimeTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650

    Based off:
     - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
    lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].

    Notable differences:
     - allows for variable length input frames (<= num_frames)
     - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
     - different attention block mechanism
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, patch_drop_rate=0.0,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 num_frames=8, time_init='rand', freeze_first_frame=False, attention_style='frozen-in-time', clip=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input
            time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
                        as ViT.
            attention_style: (str) how to attend to space and time.
            clip: (bool) use openai's CLIP instead of ImageNet21k init.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.patch_drop_rate = patch_drop_rate
        self.freeze_first_frame = freeze_first_frame
        if self.freeze_first_frame:
            self.attention_style = "freeze-first-frame"
        else:
            self.attention_style = attention_style

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if clip:
            norm_layer = partial(nn.LayerNorm, eps=1e-5, elementwise_affine=True)
        print("######USING ATTENTION STYLE: ", self.attention_style)

        if hybrid_backbone is not None:
            raise NotImplementedError('hybrid backbone not implemented')
        else:
            self.patch_embed = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames)
        num_patches = self.patch_embed.num_patches

        self.patches_per_frame = num_patches // num_frames
        self.patches_per_frame_after_dropout = int(self.patches_per_frame * (1 - self.patch_drop_rate))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1,
                        embed_dim))  # remember to take pos_embed[1:] for tiling over time
        if freeze_first_frame:
            self.first_frame_temporal_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
            self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames - 1, embed_dim))
        else:
            self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.pos_drop = PatchDropout(p=patch_drop_rate, sampling='tubelet_uniform', tokens_per_frame=self.patches_per_frame, num_frames=num_frames)

        if clip:
            self.norm_pre = norm_layer(embed_dim)
            act_layer = QuickGELU
        else:
            self.norm_pre = nn.Identity()
            act_layer = nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        

        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
                attention_style=self.attention_style, act_layer=act_layer, patches_per_frame=self.patches_per_frame, 
                patches_per_frame_after_dropout=self.patches_per_frame_after_dropout)
            for i in range(depth)])
        if self.freeze_first_frame:
            self.norm = norm_layer(embed_dim)
            # TODO: Freeze this layer??
        else:
            self.norm = norm_layer(embed_dim)
            # TODO: Add optional norm layer for video path

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            print("IDENTITY")
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)

        ## einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # Update appropriate layers and model parameters
    def set_masking_ratio(self, new_masking_ratio):
        self.patch_drop_rate = new_masking_ratio
        self.patches_per_frame_after_dropout = int(self.patches_per_frame * (1 - self.patch_drop_rate))
        self.pos_drop = PatchDropout(p=self.patch_drop_rate, sampling='tubelet_uniform', tokens_per_frame=self.patches_per_frame, num_frames=self.num_frames)

    def get_num_layer(self, var_name=""):
        if var_name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("rel_pos_bias"):
            return len(self.blocks) - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return len(self.blocks)
    
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        b, num_frames, channels, _, _ = x.shape # b 101
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(2, 1)
        x = x.reshape(b, -1, self.patch_embed.embed_dim)

        BF = x.shape[0]
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        # positional embed needs to be tiled for each frame (this does [1,2,3] --> [1,2,3,1,2,3]...)
        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = self.pos_embed[:, 1:, :].repeat(1, self.num_frames, 1)
        # temporal embed needs to be repeated within each frame (this does [1,2,3] --> [1,1,1,2,2,2,3,3,3]...)
        if self.freeze_first_frame:
            temporal_embed = torch.cat([self.first_frame_temporal_embed, self.temporal_embed], dim=1)
        else:
            temporal_embed = self.temporal_embed
        tile_temporal_embed = temporal_embed.repeat_interleave(self.patches_per_frame, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)

        curr_patches = x.shape[1]
        x = x + total_pos_embed[:, :curr_patches]
        x = self.pos_drop(x)
        x = self.norm_pre(x)
        if self.training:
            n = self.patches_per_frame_after_dropout # account for patch dropout
        else:
            n = self.patches_per_frame # use all patches at inference

        f = num_frames
        #import pdb; pdb.set_trace()
        for blk in self.blocks:
            x = blk(x, self.einops_from_space, self.einops_to_space, self.einops_from_time,
                    self.einops_to_time,
                    time_n=n, space_f=f)

        #x = self.norm(x)[:, 0]
        #x = self.pre_logits(x)

        return x

    def forward(self, x):
        # convert image to video format before sending in
        if len(x.shape) == 4:
            x = x.unsqueeze(1) # convert b, c, h, w --> b, 1, c, h, w (t=1 here)
        x = self.forward_features(x)
        return x


def create_eva_vit_g_video(img_size=224,drop_path_rate=0.4,use_checkpoint=False,precision="fp32", num_frames = 12
):
    #import pdb; pdb.set_trace()
    # TODO: Choose better number of frames
    # For now, hardcode 4
    model = SpaceTimeTransformer(
        img_size=img_size,
        patch_size=14,
        embed_dim=1408,
        depth=39,
        num_heads=1408//88,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_frames=num_frames,
        time_init='rand',
        freeze_first_frame=True,
        clip=True,
        patch_drop_rate=0.75,
    )
    model.head = nn.Identity()
    model.pre_logits = nn.Identity()
    ftr_dim = model.embed_dim
    # init with weights from eva_vit_g

    vit_checkpoint = None


    if use_checkpoint:
        print("Loading vit parameters from checkpoint: ", use_checkpoint)
        vit_checkpoint = torch.load(use_checkpoint)
    else:
        vit_model = create_eva_vit_g(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
            precision=precision,
        ) # load from eva_vit_g
        vit_checkpoint = vit_model.state_dict()


    ckpt_vals = model.load_state_dict(vit_checkpoint, strict=False)

    nn.init.zeros_(model.patch_embed.proj.bias) # TODO: Change bias to be False and add flag during init
    for block in model.blocks:
        nn.init.ones_(block.norm3.weight)
        nn.init.zeros_(block.norm3.bias)

    model.pos_embed.requires_grad = False
    model.cls_token.requires_grad = False
    for name, layer in model.named_children():
        if name == "blocks":
            for block in layer:
                for b_name, b_layer in block.named_children():
                    if b_name in CLIP_INIT_LAYERS:
                        for p in b_layer.parameters():
                            p.requires_grad = False
        elif name in CLIP_INIT_LAYERS:
            print(f"Freezing {name}")
            for p in layer.parameters():
                p.requires_grad = False
        else:
            print(f"Skipping {name} for freezing")
    model.fc = nn.Identity()
    model.vid_proj = nn.Identity()
    
    # convert to specified precision -- currently disabled do to instabilities
    #if precision == "fp16":
    #    print("Casting layers to: ", precision)
    #    model = model.half()
    
    # use checkpoint
    if use_checkpoint:
        print("Loading checkpoints is not yet supported!")
        raise NotImplementedError

    return model
