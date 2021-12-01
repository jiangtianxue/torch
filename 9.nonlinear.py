import torch
import torch.nn as nn

input = torch.tensor([  [1, -0.5],
                        [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))

print(input.shape)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid1(x)
        return x

module = MyModule()
output = module(input)
print(output)