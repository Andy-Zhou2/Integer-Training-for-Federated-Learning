import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Any
from torch.optim.lr_scheduler import StepLR
import logging
import os

from ..dataset.fp_dataset import ClientDatasetFP


def train_one_epoch(model, device, train_loader, optimizer, epoch, verbose):
    model.train()

    correct_count = 0
    total_count = 0
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        data = batch['image']
        target = batch['label']
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total_count += target.size(0)
        correct_count += (predicted == target).sum().item()

        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    if verbose:
        logging.info(
            f'Train Epoch: {epoch} Accuracy: {correct_count}/{total_count} ({100. * correct_count / total_count:.2f}%) Loss: {total_loss:.6f}')
    return total_loss, correct_count / total_count


def evaluate_model(model, device, test_loader, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data = batch['image']
            target = batch['label']
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    if verbose:
        logging.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
                     f'({100. * correct / len(test_loader.dataset):.2f}%)')
    return test_loss, accuracy


def train(model: torch.nn.Module, device: torch.device, data: ClientDatasetFP, config: Dict[str, Any]) -> Dict[
    str, Any]:
    result = {
        'train_total_loss': [],
        'test_avg_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
    }

    optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    train_loader = data['train']
    test_loader = data['test']

    logging.info('initial eval')
    test_loss, test_acc = evaluate_model(model, device, test_loader, True)

    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, epoch, config['verbose'])
        result['train_total_loss'].append(train_loss)
        result['train_accuracy'].append(train_acc)

        if config['weight_folder']:  # if '', don't save
            os.makedirs(config['weight_folder'], exist_ok=True)
            torch.save(model.state_dict(), f"{config['weight_folder']}/model_{epoch}.pt")

        if config['test_every_epoch']:
            if test_loader is None:
                raise ValueError("Test loader is None")
            test_loss, test_acc = evaluate_model(model, device, test_loader, True)
            result['test_avg_loss'].append(test_loss)
            result['test_accuracy'].append(test_acc)
        scheduler.step()
    return result
