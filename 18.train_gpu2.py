import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn


# 使用GPU训练，可以使用.to(device)方法
# 仍然是模型，损失函数和数据，其中模型和损失函数没有必要再次进行
# 赋值，需要再次赋值的只有数据，即只有 module.to(device)，loss_fn.to(device)也是正确的

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_avaiable() else cpu)

# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root = './cifar10_set',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = True
)

test_data = torchvision.datasets.CIFAR10(
    root = './cifar10_set',
    train = False,
    transform = torchvision.transforms.ToTensor(),
    download = True
)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用DatLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 构建神经网络结构
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.sequen1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64), 
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.sequen1(x)
        return x


# 实例化网络模型
module = MyModule()
module = module.to(device)

# 实例化损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 实例化优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)

# 设置训练网络的一些参数，注意区分训练次数和epoch的区别
# 训练/测试的次数是在一轮训练中分为多次数据输入训练或者测试
# 以batch_size为单位，数据集总数除以batch_size就是一轮训练中
# 的训练总次数；而epoch是对整体训练/测试集进行多少轮的训练/测试
# epoch是大的概念，训练/测试次数是小的概念

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 10


# 添加tensorboard
# writer = SummaryWriter("./logs_train")



# 训练过程
for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i+1))
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs  = module(imgs)
        loss = loss_fn(outputs, targets)

        # 优化过程，首先要梯度清零;
        # 然后反向传播;
        # 然后进行优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # 训练次数逢100才会打印出训练的次数和训练时的loss
        if total_train_step % 100 == 0:
            # tensor数据类型的item方法会去掉显示时候的tensor字符，只会显示数字
            print("训练次数：{}, loss: {}".format(total_train_step, loss.item()))
            
            # 将每一次的train loss加到tensorboard中
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试过程：跑完一轮训练之后进行一次测试，以评估训练是否有效
    # 重点看loss在测试集上是不是随着训练轮次的增加而降低
    # 这里不需要进行优化，只需要得到测试集的loss即可
    # 我们对一个模型真正的评估是在测试集上loss不断下降，
    # 训练集loss不断下降不具有说服力，要看的是测试集产生的数据

    # with torch.no_grad()使得数据梯度保持不同，自然也不会继续进行优化
    # 求得整体一个数据集上的loss，而不是记录每一个测试步骤的loss
    # 通过计算正确分类个数计算正确率
    total_test_loss = 0
    correct_num = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = module(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            correct_num += (outputs.argmax(1) == targets).sum()
    print("整体测试集上的loss: {}".format(total_test_loss))
    # 这里的correct_num是一个tensor，直接除以test_data_size这个纯数字得不到正确答案
    # 需要使用.item()方法获取到纯数字才可以
    print("整体测试集上的正确率: {}".format(correct_num.item()/test_data_size))

    # 将每一轮的test loss写到tensorboard
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", correct_num/test_data_size, total_test_step)
    # total_test_step += 1

    # 保存每一轮训练得到的模型
    torch.save(module, "./checkpoint/module_{}.pth".format(i))
    print("模型已保存")
# writer.close()