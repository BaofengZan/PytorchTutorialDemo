#coding=utf-8

"""
动态控制流程
权重共享

为了展示PyTorch的动态图的强大, 我们实现了一个非常奇异的模型: 一个全连接的
ReLU激活的神经网络, 每次前向计算时都随机选一个1到4之间的数字n, 然后接下来
就有n层隐藏层, 每个隐藏层的连接权重共享
"""

import random
import torch
from torch.autograd import Variable


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        构造函数中创建需要的层
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        对于模型的正向通道,我们随机选择0,1,2或3,
        并重复使用多次计算隐藏层表示的middle_linear模块.
        由于每个正向通道都会生成一个动态计算图,因此在定义模型的正向通道时,
        我们可以使用普通的Python控制流操作符(如循环或条件语句).
        在这里我们也看到,定义计算图时多次重复使用相同模块是完全安全的.
        这是Lua Torch的一大改进,每个模块只能使用一次.
        :param x:
        :return:
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)

        return  y_pred

# 模拟输入
N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = DynamicNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print (t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()









