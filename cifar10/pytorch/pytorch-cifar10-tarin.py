import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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

# 随机得到一些训练图片
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))

# 打印图片标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 如果你没有GPU，那么可以忽略device相关的代码
# device = torch.device("cpu:0")
# device = torch.device("cuda:0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LeNet().to(device)
# 使用多张卡并行训练
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
# net = LeNet()
# net = nn.DataParallel(net)
# net.to(device)


# CrossEntropyLoss就是我们需要的损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Start Training...")
for epoch in range(30):
    # 我们用一个变量来记录每100个batch的平均loss
    loss100 = 0.0
    # 我们的dataloader派上了用场
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 注意需要复制到GPU
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss100 += loss.item()
        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss100 / 100))
            loss100 = 0.0

print("Done Training!")

# 保存模型
# 保存整个网络
# torch.save(net, 'pytorch_cifar10_net.pkl')
# 加载模型到一个net
# loadNet = torch.load('pytorch_cifar10_net.pkl')

# 仅保存和加载模型参数(推荐使用)
torch.save(net.state_dict(), 'pytorch_cifar10_params.pkl')
net.load_state_dict(torch.load('pytorch_cifar10_params.pkl'))

# 测试准确率
# 构造测试的dataloader
dataiter = iter(testloader)
# 预测正确的数量和总数量
correct = 0
total = 0
# 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # 预测
        outputs = net(images)
        # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

# 这比随机选取（即从10个类中随机选择一个类，正确率是10%）要好很多。看来网络确实学到了一些东西。
# 那么哪些是表现好的类呢？哪些是表现的差的类呢？
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
