from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.quantization import quantize_fx


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        print(f'conv1: {x.shape}')
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
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

    # model_int8 = torch.ao.quantization.quantize_dynamic(
    #     model,  # the original model
    #     {'conv1'},  # a set of layers to dynamically quantize
    #     dtype=torch.qint8)  # the target dtype for quantized weights
    # print(model_int8)

    model.eval()

    backend = 'x86'
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
    # Prepare
    example_inputs = [sample[0] for sample in dataset][:1000]

    model_prepared = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs)
    print(model_prepared)

    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    print(model_quantized)

    def fw(x):
        self = model_quantized
        conv1_input_scale_0 = self.conv1_input_scale_0
        conv1_input_zero_point_0 = self.conv1_input_zero_point_0
        quantize_per_tensor = torch.quantize_per_tensor(x, conv1_input_scale_0, conv1_input_zero_point_0,
                                                        torch.quint8);
        x = conv1_input_scale_0 = conv1_input_zero_point_0 = None
        conv1 = self.conv1(quantize_per_tensor);
        print(f'conv1: {conv1.dtype}')
        quantize_per_tensor = None
        conv2 = self.conv2(conv1);
        conv1 = None
        max_pool2d = torch.nn.functional.max_pool2d(conv2, 2, stride=None, padding=0, dilation=1, ceil_mode=False,
                                                    return_indices=False);
        conv2 = None
        dropout1 = self.dropout1(max_pool2d);
        max_pool2d = None
        flatten = torch.flatten(dropout1, 1);
        dropout1 = None
        fc1 = self.fc1(flatten);
        flatten = None
        dropout2 = self.dropout2(fc1);
        fc1 = None
        fc2 = self.fc2(dropout2);
        dropout2 = None
        dequantize_8 = fc2.dequantize();
        fc2 = None
        log_softmax = torch.nn.functional.log_softmax(dequantize_8, dim=1, _stacklevel=3, dtype=None);
        dequantize_8 = None
        return log_softmax
    model_quantized.forward = fw

    test(model_quantized, 'cpu', test_loader)



if __name__ == '__main__':
    main()