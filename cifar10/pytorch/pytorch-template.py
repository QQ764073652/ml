import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # TODO 在这里定义网络层

    def forward(self, x):
        # TODO 在这里定义网络结构
        return x


# TODO 数据集在这里加载

# TODO 在这里编辑使用cpu或gpu设备，这里或检测gpu设备，如果没有就使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)

# TODO 在这里定义损失函数，这里使用的是交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# TODO 在这里定义优化算法，这里使用的是随机梯度下降算法
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# TODO 在这里进行迭代训练

# TODO 在这里保存模型，这里将模型参数保存在/userhome/cifar10/pytorch_cifar10_params.pkl路径下
torch.save(net.state_dict(), '/userhome/cifar10/pytorch_cifar10_params.pkl')

# TODO 在这里编辑保存到tensorboard的内容，这里保存网络结构
writer = SummaryWriter(log_dir="./logs/",comment="cifar10-cnn7")
with writer:
    writer.add_graph(net,input_to_model=torch.rand(1,3,32,32))

# TODO 在这里编写模型预测
pass