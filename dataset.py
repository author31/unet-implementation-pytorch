import os
import random
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils,models
from skimage import io, transform



class Segment(Dataset):
  def __init__(self,root,input_transform =None,train=True):
    self.root = root
    self.input_transform = input_transform
    self.train = train
    if self.train:
      self.imgs = list(sorted(os.listdir(os.path.join(root,"Images"))))
      self.masks = list(sorted(os.listdir(os.path.join(root,"Masks"))))
    else:
      self.imgs = list(sorted(os.listdir(os.path.join(root,"ValidImages"))))
      self.masks = list(sorted(os.listdir(os.path.join(root,"ValidMasks"))))
    
  def __getitem__(self,idx):
    if self.train:
        img_path = os.path.join(self.root,"Images",self.imgs[idx])
        masks_path = os.path.join(self.root,"Masks",self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask= Image.open(masks_path)
        

        if self.input_transform is not None:
            img,mask = self.transform(img,mask)
            encoded = self.encode_segmap(np.array(mask))
            img = self.input_transform(img)
            
                    
        return img, encoded
    else:
        img_path = os.path.join(self.root,"ValidImages",self.imgs[idx])
        masks_path = os.path.join(self.root,"ValidMasks",self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask= Image.open(masks_path)
        
        if self.input_transform is not None:
            img,mask = self.transform(img,mask)
            encoded = self.encode_segmap(np.array(mask))
            img = self.input_transform(img)
            
        return img, encoded

            
  def __len__(self):
    return len(self.imgs)

  def get_pascal_labels(self): 
    """Load the mapping that associates pascal classes with label colors
        Returns: np.ndarray with dimensions (21, 3)"""
    return np.asarray([[0, 0, 0],
                        [255, 255, 0]])
      
      
  def encode_segmap(self,mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(self.get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask
  
  def transform(self, image, mask):
    # Random horizontal flipping
      if random.random() > 0.5:
          image = TF.hflip(image)
          mask = TF.hflip(mask)

    # Random vertical flipping
      if random.random() > 0.5:
          image = TF.vflip(image)
          mask = TF.vflip(mask)
          
      return image,mask
    
    