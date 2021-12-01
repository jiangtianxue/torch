import torch
import torch.nn as nn
from torch.nn import MaxPool2d

input  = torch.tensor([ [1,2,0,3,1],
                        [0,1,2,3,1],
                        [5,2,3,1,1],
                        [1,2,1,0,0],
                        [2,1,0,1,1]], dtype=torch.float32)
# 因为输入必须要转变成(N,C,H,W)的四维形式才可以输入到maxpool
# 之前的input的维度是[5,5],reshape之后是[1,1,5,5]
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # 调用MaxPool2d类，括号里的两个参数也会传入到MaxPool2d的__init__方法中
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
    
    def forward(self, input):
        output = self.maxpool1(input)
        return output

module = MyModule()
output = module(input)
print(output)