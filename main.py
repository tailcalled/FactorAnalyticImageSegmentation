import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

print(torch.__version__)

BATCH_SIZE = 64

## transformations
transform = transforms.Compose(
    [transforms.ToTensor()])

## download and load training dataset
trainset = torchvision.datasets.CocoDetection(root='./data', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

import matplotlib.pyplot as plt
import numpy as np

## functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

## get some random training images
dataiter = iter(trainloader)
image, labels = dataiter.next()
print(image.shape)
print(image[0, 0, 0, 0])

blurred_img = transforms.GaussianBlur(7, 7)(image)
res_img = image - blurred_img
print(torch.cat([res_img, image], axis=1).shape)

deblur_img = transforms.GaussianBlur(99, 10)(res_img)
deblur_img_norm = transforms.GaussianBlur(99, 10)(torch.abs(res_img))
deblur_img = deblur_img / deblur_img_norm

res_img = (res_img + 1)/2

deblur_img = deblur_img + res_img
deblur_img = (deblur_img - torch.min(deblur_img)) / (torch.max(deblur_img) - torch.min(deblur_img))

## show images
#imshow(torchvision.utils.make_grid(torch.cat([deblur_img, res_img, image])))
imshow(torchvision.utils.make_grid(torch.cat([res_img, blurred_img, image])))
