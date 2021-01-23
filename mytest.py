import gdrive.MyDrive.myloader as myl
import gdrive.MyDrive.mymodel as mymodel
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

testloader = iter(myl.testloader)
images, labels = myl.dataiter.next()
device = mymodel.device

class Testing:
    
    def __init__(self):
        pass

    def dryrun(self):
        plt.imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        print(images[0].shape)
        outputs = mymodel.net(images.to(device))
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % myl.classes[predicted[j]]
                              for j in range(4)))

    def dotest(self):  
        total=0
        correct=0      
        with torch.no_grad():
            for data in myl.testloader:
                images, labels = data
                images=images.to(mymodel.device)
                labels=labels.to(mymodel.device)
                outputs = mymodel.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            
    def checkStats(self):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images=images.to(device)
                labels=labels.to(device)
                outputs = mymodel.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (myl.classes[i], 100 * class_correct[i] / class_total[i]))