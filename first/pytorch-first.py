import torch
import numpy as np

a = torch.rand(5, 3)
b = torch.rand(5, 3)
c= a+b
print(a,b,c)

c=torch.tensor([2.5])
print(c)
print(c.item())
print(a.add_(b))

# cuda
if torch.cuda.is_available():
    x = torch.rand(5,3)
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype