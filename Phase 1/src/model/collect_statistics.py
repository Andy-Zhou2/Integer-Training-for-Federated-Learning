import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNIST_Net


def main():
    torch.manual_seed(1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../data', train=True,
                             transform=transform, download=True)

    model = MNIST_Net()
    model.load_state_dict(torch.load("../data/model_ckpt/mnist_cnn.pt"))

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
    m.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(reduce_range=True),
        weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    )

    torch.quantization.prepare(m, inplace=True)

    """Calibrate
    - This example uses random data for convenience. Use representative (validation) data instead.
    """
    with torch.inference_mode():
        for i in range(150):
            x = dataset_train[i][0].unsqueeze(0)
            m(x)

    print(m)

    """Convert"""
    torch.quantization.convert(m, inplace=True)

if __name__ == '__main__':
    main()