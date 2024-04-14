import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


# Model for MNIST
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 20000)
        self.bn1 = nn.BatchNorm1d(4096)  # Batch normalization for first fully connected layer
        self.fc2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)  # Batch normalization for second fully connected layer
        self.fc3 = nn.Linear(2048, 1024)
        self.bn3 = nn.BatchNorm1d(1024)  # Batch normalization for third fully connected layer
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)  # Batch normalization for fourth fully connected layer
        self.fc5 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)   # Batch normalization for fifth fully connected layer
        self.fc6 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)   # Batch normalization for sixth fully connected layer
        self.fc7 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)   # Batch normalization for seventh fully connected layer
        self.fc8 = nn.Linear(128, 128)
        self.bn8 = nn.BatchNorm1d(128)   # Batch normalization for eighth fully connected layer
        self.fc9 = nn.Linear(128, 10)    # Output layer for 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = F.relu(self.bn8(self.fc8(x)))
        x = self.fc9(x)
        return x


model = FashionMNISTNet()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # if batch_idx % 10 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def evaluate_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)')


def main():
    # Hyperparameters
    batch_size = 20
    epochs = 100
    lr = 0.05
    gamma = 0.5  # Learning rate decay
    step_size = 10  # Learning rate decay step

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose dataset and model
    dataset = 'FashionMNIST'  # or 'MNIST' or 'FashionMNIST'
    if dataset == 'MNIST':
        model = MNISTNet().to(device)
    else:
        model = FashionMNISTNet().to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    model.apply(init_weights)

    # Data loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform) \
        if dataset == 'MNIST' else datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform) \
        if dataset == 'MNIST' else datasets.FashionMNIST('../data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        evaluate_model(model, device, test_loader)
        scheduler.step()

    print("Training complete.")


if __name__ == '__main__':
    main()
