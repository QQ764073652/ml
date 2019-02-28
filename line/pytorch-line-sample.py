import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')

x_train = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)  # x data (tensor), shape=(100, 1)
y_train = 3 * x_train + 0.2 * torch.rand(x_train.size())

# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
# y_train = torch.from_numpy(y_train).type(torch.FloatTensor)


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 1000
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(inputs)  # 前向传播
    loss = criterion(out, target)  # 计算loss

    # backward
    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch,
                                                  num_epochs,
                                                  loss.data[0]))

model.eval()
predict = model(Variable(x_train))
predict = predict.data.numpy()

import matplotlib.pyplot as plt

plt.cla()
plt.scatter(x_train.data.numpy(), y_train.data.numpy())
plt.plot(x_train.data.numpy(), predict, 'r-', lw=5)
plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
plt.show()
