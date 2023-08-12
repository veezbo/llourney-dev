import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL
from diffusers.models.embeddings import Timesteps
from diffusers.pipelines.unidiffuser.modeling_uvit import PatchEmbed
from diffusers.pipelines.pipeline_utils import numpy_to_pil

from transformers import LlamaTokenizerFast
from transformers import LlamaModel
from transformers import AutoTokenizer, AutoModel

from datasets import load_dataset

from typing import Optional

# Input image dimension
IMG_DIM = 128

# Patch dimension and latent quantities
PATCH_DIM = 16
LATENT_DIM = IMG_DIM // 8  # TODO(3): Set this dynamically from VAE config
LATENT_CHANNELS = 4
NUM_PATCHES = (LATENT_DIM // PATCH_DIM) * (LATENT_DIM // PATCH_DIM)

# Train Vars
BATCH_SIZE = 2

# Hidden embedding dimension
EMBEDDING_DIM = 768  # NOTE: Make sure EMBEDDING_DIM has the same dimensionality as C (output of this Llama checkpoint)

STABILITY_MODEL = 'runwayml/stable-diffusion-v1-5'
MODEL = 'gpt2'
TOKENIZER = MODEL
# LLAMA_MODEL = 'abhinavkulkarni/meta-llama-Llama-2-7b-chat-hf-w4-g128-awq'
# LLAMA_TOKENIZER = LLAMA_MODEL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LLourney(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(STABILITY_MODEL, subfolder='vae')
        # We use the pre-trained VAE from Stable Diffusion with frozen weights
        self.vae.requires_grad_(False)

        # self.vae_img_in = PatchEmbed(
        #     height=sample_size,
        #     width=sample_size,
        #     patch_size=patch_size,
        #     in_channels=in_channels,
        #     embed_dim=self.inner_dim,
        #     use_pos_embed=use_patch_pos_embed,
        # )

        self.vae_img_patch_embed = PatchEmbed(
            height=LATENT_DIM,
            width=LATENT_DIM,
            patch_size=PATCH_DIM,
            in_channels=LATENT_CHANNELS,
            embed_dim=EMBEDDING_DIM,
            use_pos_embed=True,
        )
        
        self.timestep_embed = Timesteps(
            EMBEDDING_DIM,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        
        # TODO(2): Use official tokenizer and model / upgrade the tokenizer and model
        # self.llama_tokenizer = LlamaTokenizerFast.from_pretrained(LLAMA_MODEL)
        # self.llama = LlamaModel.from_pretrained(LLAMA_MODEL, device_map="auto", torch_dtype=torch.float16)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        self.llama = AutoModel.from_pretrained(MODEL)

        self.llama_token_embs = self.llama.get_input_embeddings()  # TODO: May be a better way

        # LC*L*L/NP
        self.transformer_img_proj = nn.Linear(EMBEDDING_DIM, LATENT_CHANNELS*LATENT_DIM*LATENT_DIM//NUM_PATCHES)

    @torch.no_grad()
    def encode_image(self, img: Tensor) -> Tensor:
        if self.train:
            latent_image = self.vae.encode(img).latent_dist.sample()
        else:
            latent_image = self.vae.encode(img).latent_dist.mean
        return latent_image
    
    @torch.no_grad()
    def decode_image_latents(self, denoised_latent_image: Tensor, as_numpy: bool = True) -> np.ndarray:
        denoised_latent_image = 1 / self.vae.config.scaling_factor * denoised_latent_image
        image = self.vae.decode(denoised_latent_image, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        
        if as_numpy:
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        return image

    def forward(self, latent_image: Tensor, input_ids: Tensor, timestep: Tensor, text_pad_mask: Optional[Tensor] = None) -> Tensor:
        # latent_image # (B, LC, L, L) - latent image from VAE
        # input_ids (B, T): [0, VOCAB_SIZE)
        # timestep (B): [0, MAX_DIFFUSE_TIMESTEP)
        # text_pad_mask (B, T)
        B, _, latent_img_dim, _ = latent_image.shape
        assert latent_img_dim == LATENT_DIM, "Latent image dim must match pre-defined LATENT_DIM"
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
        #   so it can attend to them when understanding the patches
        # TODO(1): Likely need to pad the text to be a fixed length (i.e. a fixed portion of the concatenated embeddings)
        # TODO(1): May need some padding here to ensure dim 1 is max sequence length and valid input to llama
        # ^^ PENDING a test forward call
        transformer_emb = torch.cat([text_embs, timestep_emb, patch_embs], dim=1)  # (B, T+1+NP, C)
        # print("transformer_emb pre llama:", transformer_emb.shape)

        # Push unified embedding through Llama transformer blocks
        total_pad_mask = F.pad(text_pad_mask, (0, NUM_PATCHES+1), value=1)  # (B, T+1+NP)
        transformer_emb = self.llama(inputs_embeds=transformer_emb, attention_mask=total_pad_mask).last_hidden_state  # (B, S, C)
        # print("transformer_emb post llama:", transformer_emb.shape)

        # TODO(3): Use einops to maybe do the reshaping and shuffling in a single step while being a lot more obvious
        # Blast out the patches. Also includes a linear layer
        # NOTE: In order to undo the patching, we have a linear layer back to NUM_PATCHES and shuffle data to reform an image with same dim
        # NOTE: Recover the latent image channels from hidden dimension
        # Pluck out last NP tokens correspoding to patch embeddings
        projected_img_emb = self.transformer_img_proj(transformer_emb[:, -NUM_PATCHES:, :])  # (S[B, :-NP, C]) @ (C, LC*L*L/NP) -> (B, NP, LC*L*L/NP)
        # print("projected_img_emb after projection:", projected_img_emb.shape)
        
        # Pure reshaping to original image shape, no learnable parameters
        # (B, NP, LC*L*L/NP) -> (B, sqrt[NP], sqrt[NP], L/sqrt[NP], L/sqrt[NP], LC) | B*LC*L^2 on both sides
        # NOTE: NP = (LATENT_DIM // PATCH_DIM)^2
        projected_img_emb = projected_img_emb.reshape(
            shape=(B, LATENT_DIM // PATCH_DIM, LATENT_DIM // PATCH_DIM, PATCH_DIM, PATCH_DIM, LATENT_CHANNELS)
        )
        # print("projected_img_emb after reshape:", projected_img_emb.shape)
        # (B, H, W, P, Q, C) = (B, sqrt[NP], sqrt[NP], L/sqrt[NP], L/sqrt[NP], LC)
        # (B, LC, sqrt[NP], L/sqrt[NP], sqrt[NP], L/sqrt[NP])
        projected_img_emb = torch.einsum("bhwpqc->bchpwq", projected_img_emb)
        
        # Produce denoised image in latent space from patches
        # (B, LC, L, L)
        denoised_latent_image = projected_img_emb.reshape(
            shape=(B, LATENT_CHANNELS, LATENT_DIM, LATENT_DIM)
        )

        # NOTE: We may want to expose access to this elsewhere and/or return in output
        # Decode denoised latent image to make final prediction of noise in input image space
        # denoised_image = self.vae.decode(denoised_latent_image).sample

        return denoised_latent_image

    @torch.no_grad()
    def decode_image_latents(self, denoised_latent_image: Tensor) -> np.ndarray:
        denoised_latent_image = 1 / self.vae.config.scaling_factor * denoised_latent_image
        image = self.vae.decode(denoised_latent_image, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image


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
def forward_pass_test():
    img = torch.randn([BATCH_SIZE, 3, IMG_DIM, IMG_DIM], device=DEVICE)
    input_str = """
    Once upon a time, in the heart of an ancient kingdom, there was a stunning garden filled with vibrant flowers and chirping birds. Amidst the garden, a glistening pond served as a mirror, reflecting the blossoming surroundings. 
    A gentle breeze rustled the leaves of the towering trees, adding a melodious charm to the scene. The garden was indeed a paradise, a spectacle of nature's magnificence, where every creature lived in har
    """
    denoised_latent_img = model(img, [input_str]*BATCH_SIZE, [1]*BATCH_SIZE)
    # print(denoised_latent_img.shape)
    predicted_img = model.decode_image_latents(denoised_latent_img)

    predicted_img_pil = numpy_to_pil(predicted_img)

    for i, im in enumerate(predicted_img_pil):
        im.save(f'pred_img_{i+1}.png')


# forward_pass_test()

# Load data for training
# dataset = load_dataset("lambdalabs/pokemon-blip-captions")

# Training Loop TODO(1)
