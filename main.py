import torch
import torch.nn as nn
from model import Resunet
from dataset import Segment
from transform import decode_segmap
from transform import plotting
from transform import normalized
from torchvision import transforms, utils,models
from torch.utils.data import Dataset, DataLoader


#Composition of transformation
transformation =transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])



#initialize the dataset
root ="dataset"
dataset = Segment(root, input_transform = transformation, train=True)
testset = Segment(root, input_transform = transformation, train=False)

#initialize the dataloader
loader =DataLoader(dataset,batch_size=1,shuffle=True)
valloader =DataLoader(testset,batch_size=1,shuffle =True)


#initialize the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Resunet()
model = model.to(device)

#initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.99)

#initialze the criteriton 
criterion = nn.CrossEntropyLoss()


def train():
    epochs =0
    for img,mask in loader:
        model.train()
        optimizer.zero_grad()
        img = img.to(device,dtype=torch.float)
        mask = mask.to(device,dtype=torch.float)
        preds = model(img)
        loss =criterion(preds,mask.long())
        loss.backward()
        optimizer.step()    
        with torch.no_grad():
            for valimg,valmask in valloader:
                model.eval()
                valimg = valimg.to(device,dtype=torch.float)
                valmask = valmask.to(device,dtype=torch.float)
                valpreds = model(valimg)
                loss = criterion(valpreds,valmask.long())
                epochs +=1
            print(f"Finished: {epochs}")
            print(f"Train loss: {loss} | dice: {dice(preds,mask)}")
            print(f"Val loss: {loss} | dice: {dice(valpreds,valmask)}")
    

                
