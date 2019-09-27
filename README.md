# Unet-Implementation-Pytorch
This repo is using Unet architecture to do semantic segmentation: [Unet](https://arxiv.org/abs/1505.04597)


Using Resnet18 to do the convolutional task in an encoder section and the rest is as same as the original Unet papers 

The model was trained on customized VOC pascal dataset, the original has 20 classes for instance segmentation but this customized dataset has only 2 classes (0- for the background, 1- for the objects) for the purpose of background removing

The converting the 3 channel masks to the segmentation map and vice versa is the job of @meetshah1995, thanks @meetshah1995 for posting the code to github, that's really useful to my project

The augmentation process is also the job of mr @ptrblck from pytorch disscuss forum, thank you for your kindness





