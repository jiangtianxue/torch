import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 模型加载方式和保存方式之间需要一致，不可用混

# 保存方式1
# 第一个参数是模型名字，第二个参数是模型保存路径，后面的.pth是
# 常用的保存文件后缀，其实什么后缀都可以
# 不仅保存了网络模型结构，也保存了网络模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 加载方式1（针对保存方式1）
model = torch.load("vgg16_method1.pth")
print(model)

# 保存方式2
# 将模型的状态保存成为一种字典格式，将参数保存成字典格式
# 不保存结构，只保存参数（官方推荐）
# 并且指明保存路径
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 加载方式2（针对保存方式2）
model = torch.load("vgg16_method2.pth")
# 输出是一种字典形式
print(model)


# 真正恢复网络还需要网络模型的结构
# 然后将数据填充到这个模型的结构中
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

# 注意：
# 方式1加载自己定义的模型的时候，torch.load函数必须要
# 可见这个模型的定义（也就模型的类定义），要不就复制一份
# 这个类定义到加载模型的文件中，要不就使用import从模型
# 类定义的文件中导入这个类