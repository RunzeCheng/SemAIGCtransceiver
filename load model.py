

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

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler,DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoImageProcessor, UperNetForSemanticSegmentation


import math
import sympy as sy
from scipy.optimize import bisect
# from base64 import b64encode #image to video and save
# from Ipython.display import HTML
"""
stablediffusionPipline: uses the PNDMScheduler by default
AutoencoderKL: autoencoder model that can decode the latents into image
UNet2DConditionModel: UNet model for denoising perocess
PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler: different scheduler for interference
CLIPTextModel,CLIPTokenizer: Extract text as token
AutoImageProcessor, UperNetForSemanticSegmentation: Extract image segmentation
"""
device = 'cuda'

output.enable_custom_widget_manager()
notebook_login()



# Load the tokenizer/text encoder model for semantic extraction module.

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = text_encoder.to(device)

# Load the image Autoencoder for semantic extraction module.

vae_E = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True)
vae_E = vae_E.to(device)


# Load the Pre-trained UNet and it corresponding scheduler for semantic-generating module.

unet_E = UNet2DConditionModel.from_pretrained(
    'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=True)
unet_E = unet_E.to(device)

scheduler_E = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)
# scheduler_E = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)

# Load the Pre-trained UNet for fine-tuning module.

unet_D = UNet2DConditionModel.from_pretrained(
    'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=True)
unet_D = unet_D.to(device)


# Load the image decoder for Autoencoder decoder module.

vae_D = AutoencoderKL.from_pretrained(
    'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True)
vae_D = vae_D.to(device)

class FFschedular:
  '''Schedular for the fine-tuning module'''

  def __init__(self,timesteps, R_step, start=0.0001, end=0.02, init_gamma=0.005):
    self.timesteps = timesteps
    self.R_step = R_step
    self.start = start
    self.end = end
    self.alpha_sch = []
    self.init_gamma = init_gamma
  def linear_beta_schedule(self):
    """'test with defualt schadular'"""
    self.beta_sch = torch.linspace(self.start, self.end, self.timesteps)
    self.alpha_sch = 1 - self.beta_sch
    return self.alpha_sch

  def gamma_schedule(self):
    prod_alpha=[]
    alpha_table = self.linear_beta_schedule()
    print(alpha_table)
    alpha_table=alpha_table.tolist()
    print(alpha_table[0])
    for t in range(self.R_step):
      if t == 0:
        alpha_sqrt = 1
      else:
        alpha_sqrt = math.sqrt(alpha_table[t])
      prod_alpha.append(1)
      prod_alpha = [x * alpha_table[t] for x in prod_alpha]
      # prod_alpha = prod_alpha*alpha_table[t]
    gamma_anumber = list(range(self.R_step))

    multi_a_r = [x * math.sqrt(y) for x, y in zip(prod_alpha, gamma_anumber)]
    gamma_step = (1-(sum(prod_alpha)*math.sqrt(self.init_gamma)))/sum(multi_a_r)
    
    
    self.gamma_sch = torch.linspace(self.init_gamma, self.init_gamma+self.R_step*gamma_step, self.R_step)
    return self.gamma_sch

class SemanticProcessing:
  def __init__(self,promot,snr = 0, seedsize=32, height = 512, width = 512, guidence_scale = 7.5, Encoder_steps = 15, Decoder_steps = 5, ideal_channel=False):

    self.promot = promot
    self.snr = snr
    self.seedsize = seedsize
    self.height = height
    self.width = width
    self.guidence_scale = guidence_scale
    self.num_inference_steps = Encoder_steps + Decoder_steps
    self.start_step = Encoder_steps
    self.ideal_channel = ideal_channel
    self.Pn_2 = []
    # self.text_semantics = []
    # self.latent_1 = [] # Guassian noise
    # self.latents_2 = [] # Transmitter generated semantics
    # # self.all_latents_2 = []
    # self.latents_3 = [] # Semantics with channel noise-caused semantic noise
    # self.latents_4 = [] # Decoder fine-turned semantic
  # def ExtractingImageSemantic():

  def GenerateInitGnoise(self,batch_size=1):
    generator = torch.manual_seed(self.seedsize)
    # self.latents_1 = torch.randn(batch_size,seedsize=32)
    semantics = torch.randn((batch_size,unet_D.in_channels,self.height//8,self.width//8),generator=generator)
    semantics=semantics.to(device)
    return semantics

  def ExtractingTextSemantic(self,batch_size=1):
    # convert text to text token with condition
    text_token = tokenizer(self.promot, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
      text_sem = text_encoder(text_token.input_ids.to(device))[0]

    # generate unconditional token
    max_length = text_token.input_ids.shape[-1]
    uncon_token = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")

    with torch.no_grad():
      uncon_sem = text_encoder(uncon_token.input_ids.to(device))[0]

    # concatenates the given token sequence along a new dimension
    text_semantics = torch.cat([uncon_sem, text_sem])

    return text_semantics


  def DiffusionGenerateSemantic(self,semantics,text_semantics,encoder_denoise_step=5,return_encoder_latents=False):
    semantics = semantics.to(device)
    scheduler_E.set_timesteps(self.num_inference_steps)
    semantics = semantics * scheduler_E.init_noise_sigma
    # latent_hist=[]
    # for t in tqdm(scheduler_E.timesteps):
    for t in scheduler_E.timesteps:
      latent_model_input = torch.cat([semantics] * 2)
      latent_model_input = scheduler_E.scale_model_input(latent_model_input, t)
    # predict noise distribution
      with torch.no_grad():
        noise_pred = unet_E(latent_model_input,t,encoder_hidden_states = text_semantics).sample

      noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
      noise_pred = noise_pred_uncond + self.guidence_scale * (noise_pred_text - noise_pred_uncond)
      semantics = scheduler_E.step(noise_pred, t, semantics).prev_sample
      if t == scheduler_E.timesteps[-encoder_denoise_step]:
        encoder_semantics = semantics
      # latent_hist_2.append(self.latents_2)
    if not return_encoder_latents:
      return semantics
    else:
      return encoder_semantics
    # self.all_latents_2 = torch.cat(latent_hist_2, dim=0)
    # return self.all_latents_2


  def ChannelNoise(self,semantics,gain=1):
    #AWGN Channel
    h_latents = semantics.shape[2]
    w_latents = semantics.shape[3]
    Ps = torch.sum(torch.pow(semantics,2))/h_latents/w_latents
    Pn = Ps/(np.power(10,self.snr/10))
    self.Pn_2 = torch.sqrt(Pn).to("cpu")
    noise_channel=torch.randn_like(semantics).uniform_(0,1)*self.Pn_2

    semantics=semantics.to(device)
    scheduler_E.set_timesteps(self.num_inference_steps)
    if self.start_step > 0:
      start_timestep = scheduler_E.timesteps[self.start_step]
      start_timesteps = start_timestep.repeat(semantics.shape[0]).long()

      noise = torch.randn_like(semantics)
      semantics = scheduler_E.add_noise(semantics, noise, start_timesteps)
      if not self.ideal_channel:
        semantics = semantics + noise_channel/2
        # print("<<<<<<<<",self.snr,">>>>>>>>")
      else:
        print("<<<<<<<<Ideal Channel>>>>>>>>")
    return semantics

  def FineTuning(self, semantics, text_semantics, guidance_scale=7.5, return_all_latents=False):
    semantics = semantics.to(device)
    with autocast('cuda'):
      # for i, t in tqdm(enumerate(scheduler_E.timesteps[self.start_step:])):
      for i, t in enumerate(scheduler_E.timesteps[self.start_step:]):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([semantics] * 2)

        # predict the noise residual
        with torch.no_grad():
          noise_pred = unet_D(latent_model_input, t, encoder_hidden_states=text_semantics).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)*self.Pn_2

        # compute the previous noisy sample x_t -> x_t-1
        semantics = scheduler_E.step(noise_pred, t, semantics).prev_sample
    if not return_all_latents:
      return semantics



  def DecodeImage(self,semantics,I_names):

    semantics = 1 / 0.18215 * semantics
    names=locals()
    with torch.no_grad():
      image = vae_D.decode(semantics).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    Out_images = [Image.fromarray(image) for image in images]
    Out_images[0].save(filepath+I_names)#'image_out.png'
    Out_images[0]

  def OutputImage(self,I_names,UEoutput=False):
    image_semantic_E = self.GenerateInitGnoise()
    text_semantic_E = self.ExtractingTextSemantic()
    image_semantic_E = self.DiffusionGenerateSemantic(image_semantic_E,text_semantic_E)
    if not UEoutput:
      self.DecodeImage(image_semantic_E,I_names)
      #output image in transmitter
    # else:
    #   image_semantic_D = self.ChannelNoise(image_semantic_E)+image_semantic_E
    #   self.all_latents_2 = image_semantic_D+noise
    #   image_semantic_D = self.DiffusionGenerateSemantic(unet=unet_E)
    #   self.DecodeImage(image_semantic_D,I_names)