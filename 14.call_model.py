# 使用torch上提供的训练过的网络模型

import torch
import torchvision

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)

# 修改vgg16的网络结构
# 给vgg添加一个线性层，输入是1000，输出是10
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)
# 可以指定在classifier模组中加入一层
vgg16_false.classifier.add_module("add_linear", nn.Linear(1000, 10))
# 对于网络层直接修改
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)