from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np
from PIL import Image
from einops import rearrange
from huggingface_hub import hf_hub_download
from transformers.models.vit.modeling_vit import ViTModel
from diffusers import AutoencoderKL
from ...utils import BaseModule

def preprocess(images):
    if type(images) == list:
        images_pt = []
        for image in images:
            images_pt.append(T.ToTensor()(image) * 2 - 1)
        return torch.stack(images_pt)
    else:
        return T.ToTensor()(images).unsqueeze(0) * 2 - 1

def postprocess(latent_image, pil=True):
    latent_image = (latent_image + 1) / 2
    if pil:
        latent_image = latent_image[0].permute(1, 2, 0)
        return Image.fromarray((latent_image * 255).cpu().detach().numpy().astype(np.uint8))

class DINOLatentImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dino-vitb16"
        vae_model_name_or_path: str = 'ckpts/instruct-pix2pix'
        enable_gradient_checkpointing: bool = False
        image_size: int = 64
        num_channels: int = 4
        patch_size: int = 2
        num_hidden_layers: int = 4

    cfg: Config

    def configure(self) -> None:
        self.config = ViTModel.config_class.from_pretrained(
                hf_hub_download(
                    repo_id=self.cfg.pretrained_model_name_or_path,
                    filename="config.json",
                )
            )
        self.config.image_size = self.cfg.image_size
        self.config.num_channels = self.cfg.num_channels
        self.config.patch_size = self.cfg.patch_size
        self.config.num_hidden_layers = self.cfg.num_hidden_layers
        self.model: ViTModel = ViTModel(self.config)
        self.vae = AutoencoderKL.from_pretrained(self.cfg.vae_model_name_or_path, subfolder='vae')

        if self.cfg.enable_gradient_checkpointing:
            self.model.encoder.gradient_checkpointing = False

        self.vae.requires_grad_(False)

    def forward(self, latent_images: torch.FloatTensor = None, images: torch.FloatTensor = None) -> torch.FloatTensor:
        if latent_images is None:
            if images.ndim == 4:
                images = images.unsqueeze(1)

            batch_size, n_input_views = images.shape[:2]
            images = images * 2 - 1
            # images = (images - self.image_mean) / self.image_std
            latent_images = self.vae.encode(rearrange(images, "B N C H W -> (B N) C H W")).latent_dist.mode()
        else:
            batch_size = latent_images.shape[0]
        out = self.model(
            latent_images, interpolate_pos_encoding=True
        )
        local_features, global_features = out.last_hidden_state, out.pooler_output
        local_features = local_features.permute(0, 2, 1)
        local_features = rearrange(
            local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
        )

        return local_features

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError


# class DINOLatentImageTokenizer(BaseModule):
#     @dataclass
#     class Config(BaseModule.Config):
#         pretrained_model_name_or_path: str = "facebook/dino-vitb16"
#         enable_gradient_checkpointing: bool = False
#         image_size: int = 64
#         num_channels: int = 4
#         patch_size: int = 2

#     cfg: Config

#     def configure(self) -> None:
#         self.config = ViTModel.config_class.from_pretrained(
#                 hf_hub_download(
#                     repo_id=self.cfg.pretrained_model_name_or_path,
#                     filename="config.json",
#                 )
#             )
#         self.config.image_size = self.cfg.image_size
#         self.config.num_channels = self.cfg.num_channels
#         self.config.patch_size = self.cfg.patch_size
#         self.model: ViTModel = ViTModel(self.config)

#         if self.cfg.enable_gradient_checkpointing:
#             self.model.encoder.gradient_checkpointing = False

#     def forward(self, latent_images: torch.FloatTensor) -> torch.FloatTensor:
#         packed = False
#         batch_size = latent_images.shape[0]
#         out = self.model(
#             latent_images, interpolate_pos_encoding=True
#         )
#         local_features, global_features = out.last_hidden_state, out.pooler_output
#         local_features = local_features.permute(0, 2, 1)
#         local_features = rearrange(
#             local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
#         )
#         if packed:
#             local_features = local_features.squeeze(1)

#         return local_features

#     def detokenize(self, *args, **kwargs):
#         raise NotImplementedError