from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModelWithProjection, AutoTokenizer

from ...utils import BaseModule


class CLIPTextEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "openai/clip-vit-large-patch14"
        enable_gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        self.model: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
        )
        # self.model: CLIPTextModelWithProjection = CLIPTextModelWithProjection(
        #     CLIPTextModelWithProjection.config_class.from_pretrained(
        #         hf_hub_download(
        #             repo_id=self.cfg.pretrained_model_name_or_path,
        #             filename="config.json",
        #         )
        #     )
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path)

        # if self.cfg.enable_gradient_checkpointing:
        #     self.model.encoder.gradient_checkpointing = True


    def forward(self, texts: list, device: str) -> torch.FloatTensor:
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        for key, value in inputs.items():
            inputs[key] = value.to(device)
            
        output = self.model(**inputs)
        return output.text_embeds.unsqueeze(1)

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError


# class CLIPTextEncoder(nn.Module):
#     def __init__(self, pretrained_model_name_or_path: str = "ckpts/clip-vit-large-patch14"):
#         super().__init__()
#         self.pretrained_model_name_or_path = pretrained_model_name_or_path
#         self.model: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
#                 self.pretrained_model_name_or_path,
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

#     def forward(self, texts: list, device: str) -> torch.FloatTensor:
#         inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
#         for key, value in inputs.items():
#             inputs[key] = value.to(device)
            
#         output = self.model(**inputs)
#         return output.text_embeds.unsqueeze(1)

