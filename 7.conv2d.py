import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d

dataset = torchvision.datasets.CIFAR10(
    root = "./cifar10_set",
    train = False,
    transform = torchvision.transforms.ToTensor(),
    download = True
)

dataloader = DataLoader(dataset, batch_size=64)

class MyModule(nn.Module):
    def __init__(self):
        # 使用super，继承父类的构造函数并且子类需要重写构造函数
        super(MyModule, self).__init__()
        self.conv1 = Conv2d(in_channels = 3,
                            out_channels = 6, 
                            kernel_size = 3, 
                            stride = 1, 
                            padding = 0)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

module = MyModule()
print(module)

for data in dataloader:
    # 这里的data是列表，包含了两个元素，两个元素都是tensor
    # 所以分别赋值给了imgs和targets，imgs和targets都是tensor
    imgs, targets = data
    # 因为父类Module中定义了__call__方法，该方法
    # 调用了forward()方法，当执行 实例名(输入数据) 的时候
    # 就会自动调用__call__方法，也就是调用forward方法    
    output = module(imgs)   
    print(imgs.shape)
    print(output.shape)
