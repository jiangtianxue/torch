import torch.nn as nn 
import torch

class MyModule(nn.Module):
    def __init__(self):
        # 父类的继承
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

module = MyModule()
x = torch.tensor(1.0)
y = module(x)
print(y)