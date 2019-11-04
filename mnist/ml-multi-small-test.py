from __future__ import division, print_function

import argparse
import time

import torch
import torch.nn.functional as F
from torch import distributed, nn


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False


class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    def update(self, output, target):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = self.train()
            end = time.time()
            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'time: {}'.format(end - start)
            )

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()
        for i in range(10000):
            data = torch.randn(64, 1, 28, 28)
            target = torch.randint(0, 9, (64,)).long()
            # 创建tensor
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)

        return train_loss, train_acc


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model = Net()
    if distributed_is_initialized():
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        model = nn.DataParallel(model)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(model, optimizer, None, None, device)
    trainer.fit(args.epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        distributed.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )

    run(args)


if __name__ == '__main__':
    main()

'''
python main.py --backend nccl --init-method tcp://172.19.40.4:23456 --world-size 2 --rank 0
python main.py --backend nccl --init-method tcp://172.19.40.4:23456 --world-size 2 --rank 1
'''
