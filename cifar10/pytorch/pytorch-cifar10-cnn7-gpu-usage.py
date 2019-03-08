
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
class LeNet(nn.Module):
    # 一般在__init__中定义网络需要的操作算子，比如卷积、全连接算子等等
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 最终有10类，所以最后一个全连接层输出数量是10
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    # forward这个函数定义了前向传播的运算，只需要像写普通的python算数运算那样就可以了
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 下面这步把二维特征图变为一维，这样全连接层才能处理
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# cifar-10官方提供的数据集是用numpy array存储的
# 下面这个transform会把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 在构建数据集的时候指定transform，就会应用我们定义好的transform
# root是存储数据的文件夹，download=True指定如果数据不存在先下载数据
cifar_train = torchvision.datasets.CIFAR10(root='/userhome/dataset', train=True,
                                           download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='/userhome/dataset', train=False,
                                          transform=transform)
trainloader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=True)

# 如果你没有GPU，那么可以忽略device相关的代码
# device = torch.device("cpu:0")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = LeNet().to(device)

# CrossEntropyLoss就是我们需要的损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''model size'''
type_size=4
para = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Model {} : params: {:4f}M'.format(net._get_name(), para * type_size / 1000 / 1000))

# print("Start Training...")
# for epoch in range(30):
#     # 我们用一个变量来记录每100个batch的平均loss
#     loss100 = 0.0
#     # 我们的dataloader派上了用场
#     for i, data in enumerate(trainloader):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)  # 注意需要复制到GPU
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         loss100 += loss.item()
#         if i % 100 == 99:
#             print('[Epoch %d, Batch %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, loss100 / 100))
#             loss100 = 0.0
for i, data in enumerate(trainloader):
    inputs, labels = data
    break

mods = list(net.modules())
out_sizes = []
for i in range(1, len(mods)):
    m = mods[i]
    # 注意这里，如果relu激活函数是inplace则不用计算
    if isinstance(m, nn.ReLU):
        if m.inplace:
            continue
    out = m(inputs)
    out_sizes.append(np.array(out.size()))
    print(np.array(out.size()))
    inputs = out

total_nums = 0
for i in range(len(out_sizes)):
    s = out_sizes[i]
    nums = np.prod(np.array(s))
    total_nums += nums
# 打印两种，只有 forward 和 foreward、backward的情况
print('Model {} : intermedite variables: {:3f} M (without backward)'
        .format(net._get_name(), total_nums * type_size / 1000 / 1000))
print('Model {} : intermedite variables: {:3f} M (with backward)'
        .format(net._get_name(), total_nums * type_size*2 / 1000 / 1000))