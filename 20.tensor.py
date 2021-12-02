import torch



# 创建方式
a = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32)
b = torch.Tensor(2,3)
print(b)

# 查看类型
print(a.type())

# 查看维度
print(a.shape)
print(a.size())
print(a.dim())
print(a.numel)

# 查看值，但是只可以对于一个元素的tensor操作
print(a[0][1])
print(a[0][1].item())