import numpy as np
import torch
from math import log2,e
import math
import matplotlib.pyplot as plt
import random

from load_model import SemanticProcessing

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from huggingface_hub import hf_hub_download
#Computional resource setting Trans GPU GTX A6000, Rece GPU GTX 1080 
# Cspeed_T = 1455 #MHz= 10951000000 cycles/s 
# corenumber_T=10752
# Cspeed_R = 1320
# corenumber_R=3584
# #Communication resource setting 
# Bandwidth=20 #Mb
# Channel_condition = 0 #dB
# k_bs = 10**(-2)#path loss constant
# e_bs = 4#path loss exponent
# p_bs = 40000 #40W=40000mW#transmission power
# d_bs2r = 50 #m distance
# sigma_n = -174 #dBm Signal dBm- Noise dBm=SINR dB
# snr=10 #dB


# file_size = 512*512*3#bit
# File_size_s=64*64*4
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Function to load the LAION-AI aesthetic predictor model
def load_model():
    model_id = "LAION-AI/aesthetic-predictor"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

# Function to load and preprocess the image
def preprocess_image(image_path, processor):
    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image using the CLIP processor
    inputs = processor(images=image, return_tensors="pt")
    return inputs

# Function to predict the aesthetic score of an image
def predict_aesthetic_score(image_path):
    # Load model and processor
    model, processor = load_model()
    
    # Preprocess the image
    inputs = preprocess_image(image_path, processor)
    
    # Perform inference to get the aesthetic score
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Convert output to numpy and return the score
    aesthetic_score = outputs.cpu().numpy()[0]
    return aesthetic_score

class SCenvironment():
    def __init__(self,action,state,Trans_resource, Rece_resource,Bandwidth,snr_db,latency_high,rece_diff):
        self.Trans_resource=Trans_resource
        self.Rece_resource=Rece_resource
        # self.Channel_condition=Channel_condition
        self.Bandwidth=Bandwidth
        self.action=action
        self.latency_high=latency_high
        self.snr_db=snr_db
        self.rece_diff=rece_diff
        # self.episode=episode
        self.state=state
        self.latency=[]
        

    def wgn(self,latents,snr):
        h_latents=latents.shape[2]
        w_latents=latents.shape[3]
        Ps=torch.sum(torch.pow(latents,2))/h_latents/w_latents
        Pn=Ps/(np.power(10,snr/10))
        noise=torch.randn(latents.shape).uniform_(0,1)*torch.sqrt(Pn)
        return noise

    def bit_error_rate(self,snr_db):
        snr = 10 ** (snr_db / 10.0)  # Convert SNR from dB to linear scale
        ber = 0.5 * np.exp(-snr)  # Simplified formula for BPSK modulation
        return ber

    def transLatencyCal(self,semantic_size=64*64*4):
        #SINR Calculate 
        # snr = 10 * np.log10(p_bs) * k_bs * (d_bs2r ** (-e_bs)) - sigma_n 
        snr = 10 ** (self.snr_db / 10.0) 
        t_rate = self.Bandwidth * log2(1 + snr)
        latency_tran=semantic_size*8/t_rate/1000000
        return t_rate, latency_tran
    
    def CPL(self,density_process,file_size,para_rate,corenumber,Cspeed):
        latency_com=(1-para_rate+(para_rate/corenumber))*density_process*file_size/Cspeed
        return latency_com
    
    def computLatencyCal(self,semantic_size=64*64*4,file_size=512*512*3,para_rate=1,corenumber=1):
        max_loop = 3
        L_ext = 0
        L_Unet_t = 0
        L_Unet_r = 0
        L_rec = 0
        image_path = "out"+str(self.snr_db)+str(self.Trans_resource)+str(self.Rece_resource)+str(self.action)+str(self.Bandwidth)+".png"
        ber = self.bit_error_rate(self.snr_db)
        # P_redo = 1/(1+math.e**(0.5*self.rece_diff*(1-ber)))
        self.image_generate()
        aes_score = self.predict_aesthetic_score(self,image_path)

        for i in range(max_loop):
            L_ext += L_ext
            L_Unet_t += L_ext
            L_Unet_r += L_ext
            L_rec += L_ext
        
            density_process_e = 2838240*2/file_size #cycles1095*6912*1.5*0.5*0.5
            density_process_u = 756864*2/semantic_size #cycles 1095*6912*0.4*0.5*0.5
            density_process_r = 2459808*2/semantic_size#cycles1095*6912*1.3*0.5*0.5
            L_ext = self.CPL(density_process_e,file_size,corenumber,para_rate,self.Trans_resource)
            L_Unet_t = self.CPL(density_process_u,semantic_size,corenumber,para_rate,self.Trans_resource)*self.action
            L_Unet_r = self.CPL(density_process_u,semantic_size,corenumber,para_rate,self.Rece_resource)*(self.rece_diff)
            L_rec = self.CPL(density_process_r,semantic_size,corenumber,para_rate,self.Rece_resource)

            P_seed = random.uniform(0,1)
            if aes_score >= 5:
                return L_ext,L_Unet_t,L_Unet_r,L_rec
            
            elif i == max_loop-1:
                return L_ext,L_Unet_t,L_Unet_r,L_rec

            # latency_com=(1-para_rate+(para_rate/corenumber))*density_process*file_size/Cspeed
        
    def latencyCal(self):
        
        self.latency=np.sum(self.computLatencyCal())+self.transLatencyCal()[1]
    # Generate a range of SNR values in dB
        return self.latency

    def image_generate(self):
        SemAIGC = SemanticProcessing("A cute furry cat",6, seedsize=1, Encoder_steps = self.action, Decoder_steps = 20-self.action,) #28 #30

        Image_semantics_0 = SemAIGC.GenerateInitGnoise() #Generated initial Guassian noise

        text_semantics = SemAIGC.ExtractingTextSemantic() #Text semantic information


        Image_semantic_1 = SemAIGC.DiffusionGenerateSemantic(Image_semantics_0,text_semantics) #Image semantic information


        Image_semantic_noise = SemAIGC.ChannelNoise(Image_semantic_1,gain=10)

        Image_semantic_D = SemAIGC.FineTuning(Image_semantic_noise,text_semantics)

        Image_D = SemAIGC.DecodeImage(Image_semantic_D,"out"+str(self.snr_db)+str(self.Trans_resource)+str(self.Rece_resource)+str(self.action)+str(self.Bandwidth)+".png")

    def predict_aesthetic_score(self,image_path):
        # Load model and processor
        model, processor = load_model()
        
        # Preprocess the image
        inputs = preprocess_image(image_path, processor)
        
        # Perform inference to get the aesthetic score
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        # Convert output to numpy and return the score
        aesthetic_score = outputs.cpu().numpy()[0]
        return aesthetic_score


    def Reward(self):
        self.latencyCal()
        if self.latency < self.latency_high:
            reward=1
        elif self.latency >self.latency_high*2:
            reward=0
        else:
            reward=(self.latency-self.latency_high)/(self.latency_high)
        return reward
    
        # if self.latency < self.latency_high:
        #     reward=(self.latency-self.latency_high)/(self.latency_high)
        # elif self.latency >self.latency_high*2:
        #     reward=0
        # else:
        #     reward=(self.latency-self.latency_high)/(self.latency_high)
        # return reward

    def State(self):#SNR
        self.state=np.zeros(6)
        dis_rate=0.98
        self.state[0]=self.Trans_resource/(1095*6912*1)
        self.state[1]=self.Rece_resource/(1607*2560)
        self.state[2]=(self.Bandwidth-5)/15
        self.state[3]=self.snr_db/15
        self.state[4]=(self.latency_high-5)/20
        self.state[5]=self.rece_diff/20
        return self.state


# def histBand(current_SNR,hist_SNR,dis_rate):#Bandwidth
#     ave_SNR=current_SNR*dis_rate+hist_SNR*(1-dis_rate)
#     return ave_SNR
# def histCP():#computing power

# def histLR():#latency requirement
