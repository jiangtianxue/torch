# 使用已经训练好的模型输入数据进行测试
import torchvision
import torch
from torch import nn
from PIL import Image
image_path = "./test_pic/pic3.jpg"
image = Image.open(image_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()])

image = transform(image)


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


# 测试重点是加载已经训练好的模型，并且向里面加载数据
# 如果是GPU训练的模型，使用CPU进行测试的时候，需要加上 map_laction=torch.device('cpu')
# model = torch.load("./module_save/module9.pth", map_laction=torch.device('cpu'))
model = torch.load("./checkpoint/module_9.pth")

# 训练过程中是有batch_size的，所以这里要变成四维的张量
image = torch.reshape(image, (1,3,32,32)).cuda()
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
