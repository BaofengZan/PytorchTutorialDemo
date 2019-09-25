#coding=utf-8

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print("Download data....")

trainset = torchvision.datasets.CIFAR10(root="./root", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root="./root", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)

classes = ("plane", "car", "bird", 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 展示图像
def imshow(img):
    img = img / 2 + 0.5 # 非标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

dataiter = iter(trainloader)
images, labels = dataiter.next()


imshow(torchvision.utils.make_grid(images))
#输出类别
print(" ".join('%5s' % (classes[labels[j]] for j in range(4))))
# 只有这样才显示
plt.show()

