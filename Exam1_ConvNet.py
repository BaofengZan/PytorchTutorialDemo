#coding=utf-8

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

'''
所有网络都继承于 nn.Module
在构造函数中 声明你想要的使用的所有层
在forward函数，你可以定义模型从输入到输出将如何运行
'''

class MNISTConvNet(nn.Module):
    def __init__(self):
        # 这是你实例化所有模块的地方
        # 你可以根据稍后使用你在此给出的相同名称访问他们
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50) # 20*4*4
        self.fc2 = nn.Linear(50, 10)

    # forward函数，接受一个输入网络结构，
    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # 列数不限制
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    

net = MNISTConvNet()
print(net)
'''
torch.nn 只支持mini-batches ， 不支持输入单个样本
对于单个样本使用 unsqueeze增加维度
'''
# 定义一个虚拟的输入
input = Variable(torch.randn(1, 1, 28, 28))
out = net(input)
print(out.size())
'''
ConvNet的out是一个Variable 用来计算损失
'''
#定义一个虚拟的目标 label
target = Variable(torch.LongTensor([3]))
loss_fn = nn.CrossEntropyLoss() 
err = loss_fn(out, target) #

err.backward() # 调用backward方法，会通过ConvNet将梯度传播到它的权重
print(err)


# 来访问单个层的权重和梯度
print(u"第一个卷基层的权重大小：", net.conv1.weight.grad.size())
print(net.conv1.weight.data.norm()) # 权重的norm
print(net.conv1.weight.grad.data.norm()) #梯度的norm

'''
如何检查或者修改一个层的输出和grad_output呢？

主要是使用 hook

可以在Module或者一个Variable上注册一个函数，hook可以是forward hook 也可以是backward hook。
当forward被执行后，forward hook就会被执行，backward hook 将在执行backward阶段被执行
'''
# 在conv2注册一个forward hook 来打印一些信息
# 看上面类中的函数 printnorm
def printnorm(self, input, output):
	#input 是将输出打包成tuple的input
	#输出是一个Variable， output.data 是我们感兴趣的tensor
    print("Inside " + self.__class__.__name__ + "forward")
    print("***")
    #print('input: ', type(input))
    #print('input[0]: '. type(input[0]))
    #print('output: ', type(output))
    print("***")
    print("input size:", input[0].size())
    print('output size: ', output.data.size() )
    print('output norm: ', output.data.norm())
#调用 
net.conv2.register_forward_hook(printnorm) # 注册
out = net(input)


