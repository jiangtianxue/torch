# 正则化层 BatchNorm2d 
# recurrent layers
# transformer layers
# linear layers
# dropout layers 防止过拟合
# sparse layers 用于自然语言处理
# distance functions 计算两个值之间的误差
# loss functions 
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    root = './cifar10_set',
    train = False, 
    transform = torchvision.transforms.ToTensor(),
    download = True
)

dataloader = DataLoader(dataset, batch_size=64)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output
    
module = MyModule()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # 将batch_size大小的三维图片展平成为一个一维的向量
    # 64*3*32*32=196608
    # 线性层只接受一维的向量，所以这里的reshape不适用
    # 注意这里的flatten和reshape的区别，reshape不改变维度数，仍然是四维的张量
    # 而flatten直接改变维度数为1，即变成向量
    output = torch.flatten(imgs)
    print(output.shape)
    output = module(output)
    print(output.shape)