import torch
from skimage import io, transform
from torchvision import transforms, utils,models
res = models.resnet18(pretrained=True)
import torch.nn as nn
for name,params in res.named_parameters():
    params.requires_grad =False
    
#1x1 convolution (reduce number of channels)
def conserve(ins,outs):
      return nn.Sequential(nn.Conv2d(ins, outs, 1),nn.BatchNorm2d(outs), nn.ReLU(inplace=True),
                          nn.Conv2d(outs, outs, 1),nn.BatchNorm2d(outs), nn.ReLU(inplace=True))
#Tranpose 
def convtrans2d(ins,middles,outs,stride =1):
      return nn.Sequential(nn.Conv2d(ins, middles, 3, stride=stride, padding=1), nn.BatchNorm2d(middles), nn.LeakyReLU(inplace=True),
                           nn.Conv2d(middles, middles,3, stride=stride, padding=1), nn.BatchNorm2d(middles), nn.LeakyReLU(inplace=True),
                           nn.ConvTranspose2d(middles, outs, 2, 2))

class Resunet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resdown1 = nn.Sequential(*list(res.children())[0:4]) 
        self.resconv1 = conserve(64,64)
        self.resdown2 = nn.Sequential(*list(res.children())[5])
        self.resconv2 = conserve(128,128)
        self.resdown3 = nn.Sequential(*list(res.children())[6])
        self.resconv3 = conserve(256,256)
        self.resdown4 = nn.Sequential(*list(res.children())[7])
        self.resconv4 = conserve(512,512)
        self.rescenter = convtrans2d(512,1024,512)
        self.resup4 = convtrans2d(1024,512,256)
        self.resup3 = convtrans2d(512,256,128)
        self.resup2 = convtrans2d(256,128,64)
        self.resup1 = convtrans2d(128,64,32)
        self.Up = nn.Upsample(scale_factor=2,mode="bilinear")
        self.final  = nn.Conv2d(32,2,1)
        
    def forward(self,x):
        enc1 = self.resconv1(self.resdown1(x)) #128x128x64
        enc2 = self.resconv2(self.resdown2(enc1)) #64x64x128
        enc3 = self.resconv3(self.resdown3(enc2)) #32x32x256
        enc4 = self.resconv4(self.resdown4(enc3)) #16x16x512
        center = self.rescenter(enc4) #32x32x512
        dec4 = self.resup4(torch.cat([center,self.Up(enc4)], dim=1)) #64x64x256
        dec3 = self.resup3(torch.cat([dec4,self.Up(enc3)], dim=1)) #128x128x256
        dec2 = self.resup2(torch.cat([dec3,self.Up(enc2)], dim=1)) #256x256x64
        dec1 = self.resup1(torch.cat([dec2,self.Up(enc1)], dim=1)) #512x512x32
        final = self.final(dec1) #512x512x2
        return final