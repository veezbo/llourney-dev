from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from datasets import load_dataset
from diffusers import AutoencoderKL
from diffusers.models.embeddings import Timesteps
from diffusers.pipelines.unidiffuser.modeling_uvit import PatchEmbed
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from transformers import AutoTokenizer, AutoModel


class LLourney(nn.Module):
    
    def __init__(
        self,
        image_dim: int = 256,
        patch_dim: int = 2,
        context_size: int = 32,
        vae_model_id_or_path: str = 'runwayml/stable-diffusion-v1-5',
        llm_model_id_or_path: str = 'gpt2',
        llm_tokenizer_model_id_or_path: str = 'gpt2',
    ):
        super().__init__()
        self.image_dim = image_dim
        self.patch_dim = patch_dim
        self.context_size = context_size

        self.vae = AutoencoderKL.from_pretrained(vae_model_id_or_path, subfolder='vae')
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_latent_channels = self.vae.config.latent_channels
        self.latent_dim = image_dim // self.vae_scale_factor

        self.num_patches = (self.latent_dim // self.patch_dim) ** 2

        # TODO(2): Use official tokenizer and model / upgrade the tokenizer and model
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_model_id_or_path)
        # If the tokenizer doesn't have a pad_token, set it to eos_token
        if self.llama_tokenizer.pad_token is None:
            print(f"Setting tokenizer pad_token to eos_token: {self.llama_tokenizer.eos_token}")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        # Set padding_side explicitly
        # TODO(3): try padding_side = "left"
        self.llama_tokenizer.padding_side = "right"

        self.llama = AutoModel.from_pretrained(llm_model_id_or_path)
        self.embed_dim = self.llama.config.n_embd

        # self.vae_img_in = PatchEmbed(
        #     height=sample_size,
        #     width=sample_size,
        #     patch_size=patch_size,
        #     in_channels=in_channels,
        #     embed_dim=self.inner_dim,
        #     use_pos_embed=use_patch_pos_embed,
        # )

        self.vae_img_patch_embed = PatchEmbed(
            height=self.latent_dim,
            width=self.latent_dim,
            patch_size=self.patch_dim,
            in_channels=self.vae_latent_channels,
            embed_dim=self.embed_dim,
            use_pos_embed=True,
        )
        
        self.timestep_embed = Timesteps(
            self.embed_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        self.llama_token_embs = self.llama.get_input_embeddings()  # TODO: May be a better way

        # LC*L*L/NP
        self.transformer_img_proj = nn.Linear(
            self.embed_dim, self.vae_latent_channels * self.latent_dim * self.latent_dim // self.num_patches)
    
    def tokenize(self, text, device="cpu", padding="max_length", truncation=True, **tokenizer_kwargs):
        tokenizer_output = self.llama_tokenizer(
            text, padding=padding, truncation=truncation, max_length=self.context_size, **tokenizer_kwargs
        )

        # For now only return input_ids and attention_mask
        input_ids = torch.tensor(tokenizer_output["input_ids"], device=device, dtype=torch.long)
        attention_mask = torch.tensor(tokenizer_output["attention_mask"], device=device, dtype=torch.float)

        return input_ids, attention_mask

    def encode_image(
        self,
        img: torch.FloatTensor,
        scale_latents: bool = False,
        generator: Optional[torch.Generator] = None
    ) -> torch.FloatTensor:
        if self.train:
            latent_image = self.vae.encode(img).latent_dist.sample(generator=generator)
        else:
            latent_image = self.vae.encode(img).latent_dist.mean
        if scale_latents:
            latent_image = latent_image * self.vae.config.scaling_factor
        return latent_image
    
    def decode_image_latents(self, denoised_latent_image: torch.FloatTensor, as_numpy: bool = True) -> np.ndarray:
        denoised_latent_image = 1 / self.vae.config.scaling_factor * denoised_latent_image
        image = self.vae.decode(denoised_latent_image, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        if as_numpy:
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        return image

    def forward(
        self,
        latent_image: torch.FloatTensor,
        input_ids: torch.IntTensor,
        timestep: torch.IntTensor,
        text_pad_mask: Optional[torch.IntTensor] = None,
        device: Union[str, torch.device] = None,
    ) -> torch.FloatTensor:
        # latent_image # (B, LC, L, L) - latent image from VAE
        # input_ids (B, T): [0, VOCAB_SIZE)
        # timestep (B): [0, MAX_DIFFUSE_TIMESTEP)
        # text_pad_mask (B, T)
        B, _, latent_img_dim, _ = latent_image.shape
        assert latent_img_dim == self.latent_dim, "Latent image dim must match pre-defined LATENT_DIM"
        assert B == input_ids.shape[0], "Batch dim in img must match length of input texts"
        assert B == timestep.shape[0], "Batch dim in img must match length of timesteps"
        
        # NOTE: We may want to expose access to this elsewhere and/or return in output
        # latent_image = self.vae.encode(img).latent_dist.sample()  # (B, LC, L, L)
        # print("latent_image:", latent_image.shape)
        # L = LATENT_DIM
        # LC = LATENT_CHANNELS

        # L/(Patch Dim)^2 = Num Patches = NP
        # C contains three channels worth of info blown up to hidden embedding size (C)
        patch_embs = self.vae_img_patch_embed(latent_image)  # (B, NP, C)
        # print("patch_embs:", patch_embs.shape)

        # Embedding to encode the diffusion timestep
        timestep_emb = self.timestep_embed(timestep).unsqueeze(dim=1)  # (B, 1, C)
        # print("timestep_emb:", timestep_emb.shape)

        # Embedding to encode the text to condition on. NOTE: Text is already tokenized
        # text_idx = torch.tensor(self.llama_tokenizer(input_text).input_ids, dtype=torch.long, device=DEVICE)  # (B, T): [0, VOCAB_SIZE)
        text_embs = self.llama_token_embs(input_ids.long())  # (B, T, C)
        # print("text_embs:", text_embs.shape)

        # Concatenate all embeddings together along sequence dimension in a unified embedding for Llama
        # Because Llama is purely causal, put the text and timestep embeddings before patches 
        # so it can attend to them when understanding the patches
        transformer_emb = torch.cat([text_embs, timestep_emb, patch_embs], dim=1)  # (B, T+1+NP, C)
        # print("transformer_emb pre llama:", transformer_emb.shape)

        # Push unified embedding through Llama transformer blocks
        total_pad_mask = F.pad(text_pad_mask, (0, self.num_patches + 1), value=1)  # (B, T+1+NP)
        transformer_emb = self.llama(inputs_embeds=transformer_emb, attention_mask=total_pad_mask).last_hidden_state  # (B, S, C)
        # print("transformer_emb post llama:", transformer_emb.shape)

        # Pluck out last self.num_patches tokens corresponding to patch embeddings and map the token embedding
        # dimension from the transformer hidden dimension to the size of each latent patch volume
        # (patch_dim * patch_dim * self.vae_latent_channels)
        projected_img_emb = self.transformer_img_proj(transformer_emb[:, -self.num_patches:, :])  # (S[B, :-NP, C]) @ (C, LC*L*L/NP) -> (B, NP, LC*L*L/NP)
        # print("projected_img_emb after projection:", projected_img_emb.shape)
        
        # Reshape the final transformer hidden states back to a latent image
        # We have a number of tokens corresponding to the number of patches (which is the number of patches
        # in the height dim times the number of patches in the width dim), and each token embedding has dim
        # corresponding to the size of each latent patch volume. Thus, we can pull out the latent channels and
        # then combine the number of patches in each spatial dimension with the patch_dim to recover the original
        # latent spatial resolution.
        denoised_latent_image = rearrange(
            projected_img_emb,
            'b (height_patches width_patches) (phd pwd c) -> b c (height_patches phd) (width_patches pwd)',
            height_patches=self.latent_dim // self.patch_dim,
            width_patches=self.latent_dim // self.patch_dim,
            phd=self.patch_dim,
            pwd=self.patch_dim,
            c=self.vae_latent_channels,
        )
        # print("projected_img_emb after reshape:", projected_img_emb.shape)
        # (B, H, W, P, Q, C) = (B, sqrt[NP], sqrt[NP], L/sqrt[NP], L/sqrt[NP], LC)
        # (B, LC, sqrt[NP], L/sqrt[NP], sqrt[NP], L/sqrt[NP])
        # projected_img_emb = torch.einsum("bhwpqc->bchpwq", projected_img_emb)
        
        # Produce denoised image in latent space from patches
        # (B, LC, L, L)
        # denoised_latent_image = projected_img_emb.reshape(
        #     shape=(B, self.vae_latent_channels, self.latent_dim, self.latent_dim)
        # )

        # NOTE: We may want to expose access to this elsewhere and/or return in output
        # Decode denoised latent image to make final prediction of noise in input image space
        # denoised_image = self.vae.decode(denoised_latent_image).sample

        return denoised_latent_image


# vae = AutoencoderKL.from_pretrained(STABILITY_MODEL, subfolder='vae')
# vae.to(DEVICE)
# print(f"Number of model parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad)}")
# input("waiting here for confirmation, press enter")

# llama = LlamaModel.from_pretrained(LLAMA_MODEL, device_map="auto", dtype=)
# llama.to(DEVICE)
# print(f"Number of model parameters: {sum(p.numel() for p in llama.parameters() if p.requires_grad)}")
# input("waiting here for confirmation, press enter")
# exit()

# model = LLourney()
# model.to(DEVICE)
# print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# Singular forward pass with fake images
# (B, 3, I, I)
@torch.no_grad()
def forward_pass_test(model: LLourney, input_str: str, batch_size: int = 2, device="cpu"):
    # Prepare random image in pixel space
    img = torch.randn((batch_size, 3, model.image_dim, model.image_dim), device=device)

    # Prepare input tokens and attention mask (for padding)
    input_str_list = [input_str] * batch_size
    input_ids, attention_mask = model.tokenize(input_str_list)

    # Prepare timesteps
    timestep = torch.ones((batch_size,), device=device, dtype=torch.long)

    # Map img to latent space
    latent_img = model.encode_image(img, scale_latents=True)

    # LLourney forward pass
    denoised_latent_img = model(latent_img, input_ids, timestep, attention_mask)

    # Map denoised latents back to pixel space
    predicted_img = model.decode_image_latents(denoised_latent_img)

    predicted_img_pil = numpy_to_pil(predicted_img)

    for i, im in enumerate(predicted_img_pil):
        im.save(f'pred_img_{i+1}.png')


if __name__ == '__main__':
    input_str = "test string"
    # Tested with random small AutoencoderKL initialized as follows:
    # torch.manual_seed(0)
    # vae = AutoencoderKL(
    #     block_out_channels=[32, 64],
    #     in_channels=3,
    #     out_channels=3,
    #     down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
    #     up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
    #     latent_channels=4,
    # )
    # vae.save_pretrained("test_vae")
    test_model = LLourney(
        image_dim=64,
        vae_model_id_or_path="test_vae",
        llm_model_id_or_path="hf-internal-testing/tiny-random-gpt2",
        llm_tokenizer_model_id_or_path="hf-internal-testing/tiny-random-gpt2",
    )

    forward_pass_test(test_model, input_str)


# forward_pass_test()

# Load data for training
# dataset = load_dataset("lambdalabs/pokemon-blip-captions")

# Training Loop TODO(1)
