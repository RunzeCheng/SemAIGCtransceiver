
import os
import cv2
import torch
import numpy as np

from torch import autocast
from tqdm.auto import tqdm
from pathlib import Path
from huggingface_hub import HfApi
from PIL import Image, ImageDraw
from diffusers.utils import load_image
from google.colab import output
from huggingface_hub import notebook_login
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoImageProcessor, UperNetForSemanticSegmentation

from loadmodel import SemanticProcessing, FFschedular
device = 'cuda'

output.enable_custom_widget_manager()
notebook_login()

# Load the tokenizer/text encoder model for semantic extraction module.

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = text_encoder.to(device)

# Load the image Autoencoder for semantic extraction module.

vae_E = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True) # fine-tuned checkpoint ''SemAIGC-diffusion-v1-001','
vae_E = vae_E.to(device)


# Load the Pre-trained UNet and it corresponding scheduler for semantic-generating module.

unet_E = UNet2DConditionModel.from_pretrained(
    'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=True)
unet_E = unet_E.to(device)

scheduler_E = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)



# Load the Pre-trained UNet for fine-tuning module.

unet_D = UNet2DConditionModel.from_pretrained(
    'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=True)
unet_D = unet_D.to(device)


# Load the image decoder for Autoencoder decoder module.

vae_D = AutoencoderKL.from_pretrained(
    'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True)
vae_D = vae_D.to(device)


SemAIGC = SemanticProcessing("A cute furry cat",6, seedsize=30, Encoder_steps = 5, Decoder_steps = 15,) #28 #30

Image_semantics_0 = SemAIGC.GenerateInitGnoise() #Generated initial Guassian noise
# print(latents_0)
text_semantics = SemAIGC.ExtractingTextSemantic() #Text semantic information


Image_semantic_1 = SemAIGC.DiffusionGenerateSemantic(Image_semantics_0,text_semantics) #Image semantic information

Image_semantic_1_en = SemAIGC.DiffusionGenerateSemantic(Image_semantics_0,text_semantics,return_encoder_latents=True )

Image_i = SemAIGC.DecodeImage(Image_semantics_0,"out_0.png")

Image_E = SemAIGC.DecodeImage(Image_semantic_1,"out_1.png")

Image_E_en = SemAIGC.DecodeImage(Image_semantic_1_en,"out_1_en.png")

Image_semantic_noise = SemAIGC.ChannelNoise(Image_semantic_1,gain=10)
Image_n = SemAIGC.DecodeImage(Image_semantic_noise,"out_2.png")

Image_semantic_D = SemAIGC.FineTuning(Image_semantic_noise,text_semantics)
Image_D = SemAIGC.DecodeImage(Image_semantic_D,"out_3.png")

