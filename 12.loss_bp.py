import torch
import torchvision
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import nn


inputs = torch.tensor([1,2,3], dtype=torch.float32)
targets = torch.tensor([1,2,5], dtype=torch.float32)
# 两者的shape一定要想等
print(inputs.shape, targets.shape)

loss = L1Loss()
result = loss(inputs, targets)
print(result)

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)
print(result)

loss = MSELoss()
result = loss(inputs, targets)
print(result)

loss = MSELoss()
result = loss(inputs, targets)
print(result)

inputs = torch.tensor([0.1,0.2,0.7])
targets = torch.tensor([1])
# 需要对输入变换维度，从 N 变成 (N, C), C是分类的类别数
inputs = torch.reshape(inputs, (1,3))
print(inputs)
print(inputs.shape, targets.shape)
loss = CrossEntropyLoss()
result = loss(inputs, targets)
print(result)


# loss应用到神经网络上
train_dataset = torchvision.datasets.CIFAR10(
    root = './cifar10_set',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = True
)

train_dataloader = DataLoader(train_dataset, batch_size=1)

class MyModule(nn.Module):
    # 这里的self不能忘写了
    def __init__(self):
        super(MyModule, self).__init__()
        self.sequen1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.sequen1(x)
        return x

module = MyModule()
loss = CrossEntropyLoss()

for data in train_dataloader:
    imgs, targets = data
    outputs = module(imgs)
    # print(outputs.shape, type(outputs))
    # print(targets.shape, type(targets))
    loss_rst = loss(outputs, targets)
    # print(loss_rst)
    # 反向传播就一句话，使用loss得到的数值，调用反向传播方法即可
    loss_rst.backward()