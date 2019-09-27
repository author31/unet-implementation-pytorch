import numpy as np
import torch
import matplotlib.pyplot as plt

def get_pascal_labels(): 
    """Load the mapping that associates pascal classes with label colors
        Returns: np.ndarray with dimensions (21, 3)"""
    return np.asarray([[0, 0, 0],
                        [255, 255, 0]])
      
      
def encode_segmap(mask):
  """Encode segmentation label images as pascal classes
  Args:
      mask (np.ndarray): raw segmentation label image of dimension
        (M, N, 3), in which the Pascal classes are encoded as colours.
  Returns:
      (np.ndarray): class map with dimensions (M,N), where the value at
      a given location is the integer denoting the class index.
  """
  mask = mask.astype(int)
  label_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype=np.float)
  for ii, label in enumerate(get_pascal_labels()):
      label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
  label_mask = label_mask.astype(int)
  return label_mask


def decode_segmap(label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_mask = np.array(label_mask)
        label_colours = get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, 2):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

#for showing the predicted mask (MxN)
def plotting(tensor):
  tensor = tensor[0].detach().cpu() #2x512x512
  _,preds = torch.max(tensor,dim=0) #512x512
  decode_segmap(preds,True)

#for showing the image in the dataset
def normalized(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.cpu()
    inp = inp[0].numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)