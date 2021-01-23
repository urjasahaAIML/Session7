import gdrive.MyDrive.myloader as myl
import gdrive.MyDrive.mymodel as mymodel
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

images, labels = myl.dataiter.next()
print('shape of images', images.shape)
    
class DisplayImage:
    
    def doShow(self):
        # functions to show an image
        # show images
        #imshow(torchvision.utils.make_grid(images))
        img = torchvision.utils.make_grid(images) / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # print labels
        print(' '.join('%5s' % myl.classes[labels[j]] for j in range(4)))
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    


   