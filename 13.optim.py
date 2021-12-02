import torch
import torchvision
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch import nn


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

# 所有的东西都要先定义再使用，包括数据集（train_dataset）
# 数据集加载器（train_dataloader）
# 模型（module），损失函数（loss），优化器（optim）

# 定义模型，实例化模型对象
module = MyModule()
# 定义loss，实例化loss对象
loss = CrossEntropyLoss()
# 定义优化器，实例化optim对象的时候需要传入参数
# 因为类定义的时候__init__函数需要参数
optim = torch.optim.SGD(module.parameters(), lr=0.01)

# 云服务器跑不动这么多epoch
for epoch in range(20):
    running_loss = 0.0
    # 数据集加载器的使用
    for data in train_dataloader:
        imgs, targets = data

        # 模型的使用
        outputs = module(imgs)

        # 损失函数的使用
        loss_rst = loss(outputs, targets)

        # 优化器的使用
        # 每一步都要将梯度设置为0，否则梯度会积累，计算会出错
        # 并且梯度要先设置为0，然后再进行反向传播
        optim.zero_grad()
        loss_rst.backward()
        optim.step()
        running_loss = running_loss + loss_rst
    
    # 一般都是看一轮学习下来后loss整体的变化
    print(running_loss)
