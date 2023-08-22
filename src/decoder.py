import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import RayEncoder, Transformer, SRTLinear
from transformers import PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder

class RayPredictor(nn.Module):
    def __init__(self, num_att_blocks=2, pos_octaves = 15, pos_start_octave=0, pos_end_octave=None, ray_octaves=15, ray_start_octave=0,ray_end_octave=None, out_dims=3,  input_mlp=False, output_mlp=True, latent_dim=768, mode="efficient", return_queries=False):
        super().__init__()
        
        self.return_queries=return_queries

        idim=(pos_octaves+ray_octaves)*6
        self.query_encoder = RayEncoder(pos_octaves=pos_octaves, pos_start_octave=pos_start_octave, pos_end_octave=pos_end_octave,
                                            ray_octaves=ray_octaves, ray_start_octave=ray_start_octave, ray_end_octave=ray_end_octave)
        self.transformer = Transformer(idim, depth=num_att_blocks, heads=12, dim_head=int(latent_dim/12), latent_dim = latent_dim, mlp_dim=int(latent_dim*2), selfatt=False, mode=mode)

        if input_mlp:
            self.input_mlp = nn.Sequential(
                SRTLinear(idim, 360),
                nn.ReLU(),
                SRTLinear(360, idim)
            )
        else:
            self.input_mlp = nn.Identity()
        if output_mlp:
            assert out_dims == 3 or out_dims == 128
            self.output_mlp = nn.Sequential(
                nn.Linear(idim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims)
            )   
        else:
            self.output_mlp=nn.Identity()

    def forward(self, z, x, rays):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query camera positions [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        Returns:
            output: decoded queries [batch_size, num_rays, self.out_dims]
        """
        queries = self.query_encoder(x, rays)
        queries = self.input_mlp(queries)
        transformer_output = self.transformer(queries, z)
        output = self.output_mlp(transformer_output)
        return [output, queries] if self.return_queries else output

class MixingBlock(nn.Module):
    def __init__(self, input_dim=180, slot_dim=1536, att_dim=1536, layer_norm=False):
        super().__init__()
        self.to_q = SRTLinear(input_dim, att_dim, bias=False)
        self.to_k = SRTLinear(slot_dim, att_dim, bias=False)
        if layer_norm:
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(slot_dim)

        self.scale = att_dim ** -0.5
        self.layer_norm = layer_norm

    def forward(self, x, slot_latents):
        """
        Args:
            x: query ray features [batch_size, num_rays, input_dim]
            slot_latents: slot scene representation [batch_size, num_slots, slot_dim]
        """
        if self.layer_norm:
            x = self.norm1(x)
        q = self.to_q(x)
        k = self.to_k(slot_latents)

        dots = torch.einsum('bid,bsd->bis', q, k) * self.scale
        w = dots.softmax(dim=2)  # [batch_size, num_rays, num_slots]

        # [batch_size, num_rays, num_slots, 1] * [batch_size, 1, num_slots, slot_dim]
        s = (w.unsqueeze(-1) * slot_latents.unsqueeze(1)).sum(2)

        if self.layer_norm:
            s = self.norm2(s)

        return s, w


class RenderMLP(nn.Module):
    def __init__(self, input_dim=1536+180, hidden_dim=1536, out_dim=3):
        super().__init__()
        # According to Mehdi, this uses Leaky ReLUs, and a Sigmoid at the end
        self.net = nn.Sequential(
            SRTLinear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            SRTLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            SRTLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            SRTLinear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            SRTLinear(hidden_dim, out_dim),
            nn.Identity() if out_dim==3 else nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.net(x)

class SlotMixerDecoder(nn.Module):
    """ The Slot Mixer Decoder proposed in the OSRT paper. """
    def __init__(self, num_att_blocks=2, pos_start_octave=0, layer_norm=False, mode="efficient", out_dims=3):
        super().__init__()
        self.allocation_transformer = RayPredictor(num_att_blocks=num_att_blocks,
                                                   pos_start_octave=pos_start_octave,
                                                   input_mlp=True, latent_dim=1536, mode=mode, output_mlp=False, return_queries=True)
        self.mixing_block = MixingBlock(layer_norm=layer_norm)

        self.render_mlp = RenderMLP(out_dim=out_dims)

    def forward(self, slot_latents, camera_pos, rays, **kwargs):
        x, query_rays = self.allocation_transformer(slot_latents, camera_pos, rays)
        slot_mix, slot_weights = self.mixing_block(x, slot_latents)
        pixels = self.render_mlp(torch.cat((slot_mix, query_rays), -1))
        return pixels, {'segmentation': slot_weights}

class SRTDecoder(nn.Module):
    def __init__(self, num_att_blocks=2, pos_octaves = 15, pos_start_octave=0, pos_end_octave=None, ray_octaves=15, ray_start_octave=0,ray_end_octave=None, out_dims = 3, out_mlp=True, latent_dim=768, mode="efficient"):
        super().__init__()

        idim = (pos_octaves+ray_octaves)*6+6

        self.resize_out = False
        idim = (pos_octaves+ray_octaves)*6
        self.ray_predictor = RayPredictor(num_att_blocks=num_att_blocks,
                                          pos_start_octave=pos_start_octave,
                                          pos_end_octave=pos_end_octave,
                                          pos_octaves=pos_octaves,
                                          ray_start_octave=ray_start_octave,
                                          ray_end_octave=ray_end_octave,
                                          ray_octaves=ray_octaves,
                                          latent_dim=latent_dim, 
                                          input_mlp=True, output_mlp=not out_mlp, mode=mode
                                          )
    
        if out_mlp:
            self.out_mlp = nn.Sequential(
                SRTLinear(idim, 1536),
                nn.LeakyReLU(),
                SRTLinear(1536, 1536),
                nn.LeakyReLU(),
                SRTLinear(1536, 1536),
                nn.LeakyReLU(),
                SRTLinear(1536, 1536),
                nn.LeakyReLU(),
                SRTLinear(1536, out_dims),
            )
            
        else:
            self.out_mlp= nn.Identity()
            
    def forward(self, z, x, rays, **kwargs):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query camera positions [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        Returns:
            output: decoded queries [batch_size, num_rays, self.out_dims]
        """
        output = self.ray_predictor(z, x, rays)
        output = self.out_mlp(output)

        return output

class DeFiNeDecoder(nn.Module):
    def __init__(self, pos_octaves = 20, pos_start_octave=1, pos_end_octave=30, ray_octaves=10, ray_start_octave=1,ray_end_octave=30, out_dims = 3,latent_dim=512):
        super().__init__()
        idim = (pos_octaves+ray_octaves)*6+6

        self.query_encoder = RayEncoder(pos_octaves=pos_octaves, pos_start_octave=pos_start_octave,pos_end_octave=pos_end_octave, ray_octaves=ray_octaves, ray_start_octave=ray_start_octave,ray_end_octave=ray_end_octave)
    
        config = PerceiverConfig(
            d_latents=latent_dim,
            d_model=idim,
            num_latents=2048,
            hidden_act='gelu',
            hidden_dropout_prob=0.25,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            num_blocks=1,
            num_cross_attention_heads=1,
            num_self_attends_per_block=8,
            num_self_attention_heads=8,
            qk_channels=None,
            v_channels=None,
        )
        self.transformer = PerceiverBasicDecoder(
            config,
            output_num_channels=out_dims,
            position_encoding_type="none",
            num_heads=1,
            use_query_residual=False,
            num_channels=idim
        )

    def forward(self, z, x, rays, **kwargs):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query camera positions [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        Returns:
            output: decoded queries [batch_size, num_rays, out_dims]
        """

        queries = self.query_encoder(x, rays)
        queries = torch.cat([rays, x,queries], dim=-1)
        output = self.transformer(queries,z).logits

        return output


class FeatureDecoder(nn.Module):
    def __init__(self, conv_features=128, upsample = 3, out_dims=3, norm = None, h=15, w=20):
        super().__init__()

        self.conv_blocks = nn.ModuleList([])
        self.upsample = upsample

        self.conv_0 = nn.Conv2d(conv_features, out_dims, 3, padding=1)
        for i in range(upsample):
            dims = (int(conv_features/2**(i+1)), int(h*2**(i+1)), int(w*2**(i+1)))
            self.conv_blocks.append(nn.ModuleList([
                nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=2), 
                    nn.Conv2d(int(conv_features/2**i), int(conv_features/2**(i+1)), 3, padding=1),
                    nn.BatchNorm2d(int(conv_features/2**(i+1))) if norm=="Batch" else nn.Identity(),
                    nn.LeakyReLU(0.2),
                ),
                nn.Conv2d(int(conv_features/2**(i+1)), out_dims,3, padding=1)                
            ]))

    def forward(self, x):
        """
        Args:
            x: procesed representation [batch_size x num_images, h_in, w_in, self.conv_features]
        Returns:
            output: decoded queries [batch_sizex num_images,  h_in x 2^self.upsample, w_in x 2^self.upsample, out_dims]
        """
        i = 1
        skip = F.interpolate(self.conv_0(x),scale_factor=2, mode="bilinear")
        for conv_block, conv_out in self.conv_blocks:
            
            x = conv_block(x)
            x_out = conv_out(x) + skip
            if i < self.upsample:
                skip = F.interpolate(x_out,scale_factor=2, mode="bilinear")
                i+=1

        return x_out

class RayPatchDecoder(nn.Module):
    def __init__(self, out_dims = 3, conv_features=128, h_in = 16, w_in = 16, h_out = None, w_out = None, upsample = 3, norm=None) -> None:
        super().__init__()
        self.h_in = h_in
        self.w_in = w_in
        if h_out is not None and w_out is not None:
            self.resize_out = True
            self.h_out = h_out
            self.w_out = w_out
        else:
            self.resize_out = False
        self.upsample = upsample
        
        assert conv_features/2**upsample > 1
        self.conv_decoder = FeatureDecoder(conv_features, self.upsample, out_dims, norm=norm, h=h_in, w=w_in)

    def forward(self, x):
        """
        Args:
            x: procesed representation [batch_size, num_images x self.h_in x self.w_in, self.conv_features]
        Returns:
            output: decoded images [batch_size, num_images, self.h_out, self.w_out, out_dims]. If h_out and w_out aren't set, they will be (self.h_in, self.w_in)*2^self.upsample
        """
        b, t, f = x.shape
        output = x.unsqueeze(1).reshape(-1, self.h_in, self.w_in, f)

        
        output = self.conv_decoder(output.permute(0,3,1,2))
        if self.resize_out:
            output = F.interpolate(output, (self.h_out, self.w_out))
        _, c, h, w = output.shape
        output = output.unsqueeze(0).reshape(b,-1,c,h,w).permute(0,1,3,4,2)

        return output

