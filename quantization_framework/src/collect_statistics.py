import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import MnistNet


def get_statistics(channel1=2, channel2=4, pass_quantity=150):
    """
    Get runtime statistics of the model. This is used to calibrate the quantization process.
    Requires model placed under specific directory with naming convention that agrees with the model training code.
    :param channel1: The number of channels in the first convolutional layer
    :param channel2: The number of channels in the second convolutional layer
    :param pass_quantity: The number of passes to calibrate the model
    :return: A dictionary containing the min and max values of the model's observers
    """
    torch.manual_seed(1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../data', train=True,
                             transform=transform, download=True)

    model = MnistNet(channel1, channel2)
    model.load_state_dict(torch.load(f"../model_ckpt/mnist_{channel1}_{channel2}.pt", map_location='cpu'))

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
        for i in range(pass_quantity):
            x = dataset_train[i][0].unsqueeze(0)
            m(x)

    # print(m)

    """Convert"""
    # don't convert - we don't need the converted model
    # also doesn't work for M1 mac
    # torch.quantization.convert(m, inplace=True)

    observers = dict()
    observers['input'] = m[0].activation_post_process
    observers['conv1'] = m[1].conv1.activation_post_process
    observers['conv2'] = m[1].conv2.activation_post_process
    observers['fc1'] = m[1].fc1.activation_post_process
    observers['fc2'] = m[1].fc2.activation_post_process

    results = {key: (observer.min_val.item(), observer.max_val.item()) for key, observer in observers.items()}
    return results


if __name__ == '__main__':
    r = get_statistics(channel1=2, channel2=4)
    print(r)