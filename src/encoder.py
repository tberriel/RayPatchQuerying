import torch
import torch.nn as nn
from src.layers import RayEncoder, Transformer, EfficientPerceiverEncoder, SlotAttention
from src.resnet import resnet18
from transformers import PerceiverModel, PerceiverConfig


class SRTConvBlock(nn.Module):
    """
    Based on the implementation by Karl Stelzner
    https://github.com/stelzner/srt/
    """
    def __init__(self, idim, hdim=None, odim=None, norm=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        act_0 = act_1= nn.ReLU()
        if odim is None:
            odim = 2 * hdim
        if norm =="Batch":
            norm_layer_0=nn.BatchNorm2d(hdim)
            norm_layer_1=nn.BatchNorm2d(odim)
        else:
                norm_layer_0=norm_layer_1=nn.Identity()

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            norm_layer_0,
            act_0,
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            norm_layer_1,
            act_1)
    
    def forward(self, x):
        return self.layers(x)


class SRTEncoder(nn.Module):
    """
    Based on the implementation by Karl Stelzner
    https://github.com/stelzner/srt/
    """
    def __init__(self, num_conv_blocks=4, num_att_blocks=10, pos_start_octave=0, num_octaves=15, max_pooling=False, norm=None, latent_dim=768):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=num_octaves, pos_start_octave=pos_start_octave,
                                      ray_octaves=num_octaves)
        
        idim=num_octaves*2*6+3

        conv_blocks = [SRTConvBlock(idim=idim, hdim=96, norm=norm)]
        cur_hdim = 192
        for i in range(1, num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None, norm=norm))
            cur_hdim *= 2

        self.conv_blocks = nn.Sequential(*conv_blocks)
        if max_pooling:
            self.pooling = nn.MaxPool2d(2)
        else:
            self.pooling = nn.Identity()

        self.per_patch_linear = nn.Conv2d(cur_hdim, 768, kernel_size=1)

        self.transformer = Transformer(768, depth=num_att_blocks, heads=12, dim_head=64,
                                       latent_dim=latent_dim,
                                       mlp_dim=int(latent_dim*2), selfatt=True)

    def forward(self, images, camera_pos, rays):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """
        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        ray_enc = self.ray_encoder(camera_pos, rays)
        x = torch.cat((x, ray_enc), 1)
        x = self.conv_blocks(x)
        x = self.pooling(x)
        x = self.per_patch_linear(x)
        x = x.flatten(2, 3).permute(0, 2, 1)

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        x = self.transformer(x)

        return x

class DeFiNeEncoder(nn.Module):
    def __init__(self, num_att_blocks=8, pos_start_octave=1, pos_octaves=20, pos_end_octave=30,ray_octaves=10, ray_start_octave=1, ray_end_octave=30, num_latents = 2048):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=pos_octaves, pos_start_octave=pos_start_octave,
                                        pos_end_octave=pos_end_octave, ray_octaves=ray_octaves, ray_start_octave=ray_start_octave,ray_end_octave=ray_end_octave)
        
        self.conv_blocks = resnet18()

        in_dim = 966 + (pos_octaves+ray_octaves)*6 #(512+256+128+64)+6+(pos_octaves+ray_octaves)*6

        self.config = PerceiverConfig(
            d_latents=512,
            d_model=in_dim,
            num_latents=num_latents,
            hidden_act='gelu',
            hidden_dropout_prob=0.25,#
            initializer_range=0.02,#
            layer_norm_eps=1e-12,
            num_blocks=1,
            num_cross_attention_heads=1,
            num_self_attends_per_block=num_att_blocks,
            num_self_attention_heads=8,
            qk_channels=None,
            v_channels=None
        )
        self.transformer = PerceiverModel(
            self.config,
        )

    def forward(self, images, camera_pos, rays):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """
        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)
        b, h, w, c = rays.shape
        rays = torch.nn.functional.interpolate(rays.permute(0,3,1,2),size=[int(h/4),int(w/4)], mode="bilinear").permute(0,2,3,1)
        ray_enc = self.ray_encoder(camera_pos, rays)
        ray_enc = torch.cat([rays.permute(0,3,1,2), camera_pos[...,None,None].repeat(1,1,int(h/4),int(w/4)),ray_enc], dim=1)
        
        x_skips = self.conv_blocks(x)

        b,c,h,w = x_skips[0].shape
        for i in range(1,len(x_skips)):
            x_skips[i] = torch.nn.functional.interpolate(x_skips[i],size=[h,w], mode="bilinear")
        x_skips.append(ray_enc)
        x = torch.cat(x_skips, dim=1)
        
        x = x.flatten(2, 3).permute(0, 2, 1)
        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        x = self.transformer(inputs=x).last_hidden_state

        return x


class OSRTEncoder(nn.Module):
    def __init__(self, num_conv_blocks=3, num_att_blocks=5, num_octaves=15,pos_start_octave=0, num_slots=6, slot_dim=1536, slot_iters=1, max_pooling=False, norm=None, randomize_initial_slots=False, latent_dim=768):
        super().__init__()
        self.srt_encoder = SRTEncoder(num_conv_blocks=num_conv_blocks, num_att_blocks=num_att_blocks,
                                             pos_start_octave=pos_start_octave, num_octaves=num_octaves, norm=norm, max_pooling=max_pooling, latent_dim=latent_dim)

        self.slot_attention = SlotAttention(num_slots, slot_dim=slot_dim, iters=slot_iters,
                                            randomize_initial_slots=randomize_initial_slots)

    def forward(self, images, camera_pos, rays):
        set_latents = self.srt_encoder(images, camera_pos, rays)
        slot_latents = self.slot_attention(set_latents)
        return slot_latents