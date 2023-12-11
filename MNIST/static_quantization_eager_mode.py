from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.quantization import quantize_fx
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver, PerChannelMinMaxObserver


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        # self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = self.log_softmax(x)
        return x


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    test_kwargs = {'batch_size': args.test_batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = datasets.MNIST('../data', train=True,
                              transform=transform, download=True)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    model.load_state_dict(torch.load("./model_ckpt/mnist_cnn.pt"))

    model.eval()

    """Fuse
    - Inplace fusion replaces the first module in the sequence with the fused module, and the rest with identity modules
    """
    torch.quantization.fuse_modules(model, ['conv1', 'relu1'], inplace=True)  # fuse first Conv-ReLU pair
    torch.quantization.fuse_modules(model, ['conv2', 'relu2'], inplace=True)  # fuse second Conv-ReLU pair
    torch.quantization.fuse_modules(model, ['fc1', 'relu3'], inplace=True)

    """Insert stubs"""
    m = nn.Sequential(torch.quantization.QuantStub(),
                      model,
                      torch.quantization.DeQuantStub())

    """Prepare"""
    m.qconfig = torch.quantization.get_default_qconfig('x86')
    torch.quantization.prepare(m, inplace=True)

    """Calibrate
    - This example uses random data for convenience. Use representative (validation) data instead.
    """
    with torch.inference_mode():
        for i in range(150):
            x = dataset[i][0].unsqueeze(0)
            m(x)

    """Convert"""
    torch.quantization.convert(m, inplace=True)

    test(m, 'cpu', test_loader)



if __name__ == '__main__':
    main()