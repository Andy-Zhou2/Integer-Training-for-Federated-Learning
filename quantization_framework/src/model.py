import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self, channel1, channel2):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, channel1, 3, 1)
        self.conv2 = nn.Conv2d(channel1, channel2, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(channel2 * 12 * 12, 128)  # 28x28 -> 12x12 after 2 convs and a maxpool
        self.fc2 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.log_softmax = nn.LogSoftmax(dim=0)  # set to 1 for normal batches

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool(x)
        x = self.dropout1(x)

        if x.ndim == 4:
            x = torch.flatten(x, 1)
        else:  # bs = 1, ndim = 3
            x = torch.flatten(x, 0)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        # print('x after maxpool: ', x)
        # print('maximum value:', torch.max(x))
        # # find the coord of the max value
        # print(x.shape, np.unravel_index(torch.argmax(x), x.shape))

        output = self.log_softmax(x)
        # print(f'output: {output}')
        return output


if __name__ == '__main__':
    net = MnistNet()
    net.load_state_dict(torch.load("data/model_ckpt/mnist_cnn.pt"))
    net.eval()

    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../data', train=True,
                                   transform=transform, download=True)

    net(dataset_train[0][0])
