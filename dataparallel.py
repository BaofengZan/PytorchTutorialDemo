#coding=utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# 参数和数据加载

input_size = 5
output_size = 2
batch_size = 30
data_size = 100

#伪数据集
class RandomDataset(Dataset):
    '''
    实现getitem实现生成一个随机数据集
    '''
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, 100), batch_size=batch_size, shuffle=True)

#搭建简单模型 
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("In Module: input size:", input.size(), " output size: ", output.size())

        return output

#生成一个模型的实例，并且检测是否有多个GPU 如果有，就使用nn.DataParaller 来包装我们的模型，然后就将我们的模型通过model.gpu()
# 施加于这些GPU上
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let us use, ". torch.cuda.device_count(), "GPU!")
    
    model = nn.DataParaller(model)  #包装我们的模型

if torch.cuda.is_available():
    model.cuda()

# 运行模型
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)

    output = model(input_var)
    print("outSide: input size", input_var.size(), "output_size", output.size())
    

