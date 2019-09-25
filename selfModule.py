#coding:utf-8
'''
自定义nn模块，如果已有模块串起来不能满足你的复
杂需求, 那么你就能以这种方式来定义自己的模块
'''


import torch
import torch.nn as nn
from torch.autograd import Variable

class TwoLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        '''
         构造函数中，实例化两个需要的模块，并将他们分配为成员变量
        :param D_in:
        :param D_out:
        '''
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        """
        在forward函数中,我们接受一个变量的输入数据,我们必须返回一个变量的输出数据.
        我们可以使用构造函数中定义的模块以及变量上的任意运算符.
        :param x:
        :return:
        """
        h_relu = self.linear1(x).clamp(min=0) # 截断
        y_pred = self.linear2(h_relu)
        return y_pred


N, D_in, H, D_out= 64, 1000, 100, 10
#创建随机张量来保存输入输出，
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# 实例化类
model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print (t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()






















