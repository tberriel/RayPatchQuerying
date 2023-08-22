import torch
import torch.nn as nn
import torch.nn.init as init
import math
from einops import rearrange, repeat

class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=8, start_octave=0, end_octave=None):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave
        self.end_octave = end_octave

    def forward(self, coords, rays=None):
        embed_fns = []
        batch_size, num_points, dim = coords.shape
        if self.end_octave is not None:
            octaves = torch.linspace(self.start_octave, self.end_octave, self.num_octaves, device=coords.device, dtype=torch.float32)
            multipliers = octaves * math.pi
        else:
            octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves, device=coords.device, dtype=torch.float32)
            multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayEncoder(nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0, pos_end_octave=None, ray_octaves=4, ray_start_octave=0, ray_end_octave=None):
        super().__init__()
        self.pos_encoding = PositionalEncoding(num_octaves=pos_octaves, start_octave=pos_start_octave, end_octave=pos_end_octave)
        self.ray_encoding = PositionalEncoding(num_octaves=ray_octaves, start_octave=ray_start_octave, end_octave=ray_end_octave)

    def forward(self, pos, rays):
        if len(rays.shape) == 4:
            # When the function is called by the encoder rays is [n_img, h, w, 3]
            batchsize, height, width, dims = rays.shape
            # When the input is 3D position of the camera for all points, pos will be [n_img, 3]
            pos_enc = self.pos_encoding(pos.unsqueeze(1))
            pos_enc = pos_enc.view(batchsize, pos_enc.shape[-1], 1, 1)
            pos_enc = pos_enc.repeat(1, 1, height, width)

            rays = rays.flatten(1, 2)

            ray_enc = self.ray_encoding(rays)
            ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
            ray_enc = ray_enc.permute((0, 3, 1, 2))
            x = torch.cat((pos_enc, ray_enc), 1)
        else:
            # When the function is called by the decoder rays and pos are [n_img, h*w, 3]
            pos_enc = self.pos_encoding(pos)
            ray_enc = self.ray_encoding(rays)
            x = torch.cat((pos_enc, ray_enc), -1)
        return x

# Transformer implementation based on ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class PreNorm(nn.Module):
    def __init__(self, dim, fn, dim_2=None, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        if dim_2 is not None:
            self.norm_2 = nn.LayerNorm(dim_2, eps=eps)
        else:
            self.norm_2 = nn.Identity()
        self.fn = fn
    def forward(self, x, **kwargs):
        if self.norm_2 is not None and "z" in kwargs.keys():
            kwargs["z"] = self.norm_2(kwargs["z"])
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            ViTLinear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ViTLinear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EfficientAttention(nn.Module):
    def __init__(self, dim, latent_dim = 768, heads=8, dim_head=64, dropout=0., selfatt=True, bias=False, pre_dropout=False, mode="base"):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        if mode == "efficient" and not hasattr(torch.nn.functional,"scaled_dot_product_attention"):
            print("Efficient attention only available with Pytorch 2.0. Proceeding with base attention.")
            mode = "base"
        if mode == "base":
            def attend(q,k,v):
                dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

                attn = nn.functional.softmax(dots, dim=-1)

                out = torch.matmul(attn, v)
                return out

            self.attend = lambda q,k,v: attend(q,k,v)
        elif mode =="efficient":
            self.attend = lambda q,k,v: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False) #Function available from Pytoch 2.0

        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=bias)
            self.to_kv = nn.Linear(latent_dim, inner_dim * 2, bias=bias)

        self.to_out = nn.Sequential(
            nn.Dropout(dropout) if pre_dropout else nn.Identity(),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout) if not pre_dropout else nn.Identity()
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = self.attend(q,k,v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., selfatt=True, latent_dim = 768, bias=False,eps=1e-5, pre_dropout=False, ln_dim_2=None, mode="efficient"):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, EfficientAttention(dim, latent_dim=latent_dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, selfatt=selfatt, bias=bias, pre_dropout=pre_dropout, mode=mode), eps=eps, dim_2=ln_dim_2),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout), eps=eps)
            ]))

    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x

class EfficientPerceiverEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_att_blocks,
        num_latents = 512,
        latent_dim = 512,
        num_cross_attention_heads = 1,
        num_self_attention_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        dropout_prob = 0.,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        mode = "efficient"
    ):
        super().__init__()
        self.initializer_range=initializer_range
        
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend = Transformer(latent_dim,1,num_cross_attention_heads, dim_head=cross_dim_head, mlp_dim=dim, dropout=dropout_prob, selfatt=False, latent_dim=dim, bias=True, pre_dropout=True, eps=layer_norm_eps, ln_dim_2=dim, mode=mode)

        self.self_attend = Transformer(latent_dim,num_att_blocks,num_self_attention_heads, dim_head=latent_dim_head, mlp_dim=latent_dim, dropout=dropout_prob, selfatt=True, latent_dim=latent_dim, bias=True, pre_dropout=True, eps=layer_norm_eps, mode=mode)


    def forward(
        self,
        data
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b = b)

        x = self.cross_attend(x, z=data)
        x = self.self_attend(x)

        return x
    

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif hasattr(module, "latents"):
            module.latents.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

#############################################################################
## OSRT Layers from https://github.com/stelzner/osrt/blob/main/osrt/layers.py
#############################################################################
__USE_DEFAULT_INIT__ = False


class SRTLinear(nn.Linear):
    """ Initialization for linear layers used in the SRT decoder """
    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

class JaxLinear(nn.Linear):
    """ Linear layers with initialization matching the Jax defaults """

    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            input_size = self.weight.shape[-1]
            std = math.sqrt(1/input_size)
            init.trunc_normal_(self.weight, std=std, a=-2.*std, b=2.*std)
            if self.bias is not None:
                init.zeros_(self.bias)


class ViTLinear(nn.Linear):
    """ Initialization for linear layers used by ViT """
    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.normal_(self.bias, std=1e-6)

class SlotAttention(nn.Module):
    """
    Slot Attention as introduced by Locatello et al.
    """
    def __init__(self, num_slots, input_dim=768, slot_dim=1536, hidden_dim=3072, iters=3, eps=1e-8,
                 randomize_initial_slots=False):
        super().__init__()

        self.num_slots = num_slots
        self.iters = iters
        self.scale = slot_dim ** -0.5
        self.slot_dim = slot_dim

        self.randomize_initial_slots = randomize_initial_slots
        self.initial_slots = nn.Parameter(torch.randn(num_slots, slot_dim))

        self.eps = eps

        self.to_q = JaxLinear(slot_dim, slot_dim, bias=False)
        self.to_k = JaxLinear(input_dim, slot_dim, bias=False)
        self.to_v = JaxLinear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            JaxLinear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            JaxLinear(hidden_dim, slot_dim)
        )

        self.norm_input   = nn.LayerNorm(input_dim)
        self.norm_slots   = nn.LayerNorm(slot_dim)
        self.norm_pre_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs):
        """
        Args:
            inputs: set-latent representation [batch_size, num_inputs, dim]
        """
        batch_size, num_inputs, dim = inputs.shape

        inputs = self.norm_input(inputs)
        if self.randomize_initial_slots:
            slot_means = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1)
            slots = torch.distributions.Normal(slot_means, self.embedding_stdev).rsample()
        else:
            slots = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1)

        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            norm_slots = self.norm_slots(slots)

            q = self.to_q(norm_slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            # shape: [batch_size, num_slots, num_inputs]
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(updates.flatten(0, 1), slots_prev.flatten(0, 1))
            slots = slots.reshape(batch_size, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))

        return slots

