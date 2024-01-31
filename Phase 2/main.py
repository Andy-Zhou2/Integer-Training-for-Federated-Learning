import numpy as np
from torchvision import datasets, transforms
from pktnn_fc import PktFc
from pktnn_mat import PktMat

transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset_train = datasets.MNIST('./data', train=True,
                               transform=transform, download=True)
dataset_test = datasets.MNIST('./data', train=False,
                              transform=transform, download=True)

num_train_samples = len(dataset_train)
num_test_samples = len(dataset_test)

dataset_train = [((data * 256).numpy().astype(np.int_), answer) for data, answer in dataset_train]
dataset_test = [((data * 256).numpy().astype(np.int_), answer) for data, answer in dataset_test]

mnist_train_labels = PktMat(num_train_samples, 1)
mnist_train_images = PktMat(num_train_samples, 28 * 28)
mnist_test_labels = PktMat(num_test_samples, 1)
mnist_test_images = PktMat(num_test_samples, 28 * 28)
for i in range(num_train_samples):
    mnist_train_labels[i][0] = dataset_train[i][1]
    mnist_train_images[i] = dataset_train[i][0].flatten()
for i in range(num_test_samples):
    mnist_test_labels[i][0] = dataset_test[i][1]
    mnist_test_images[i] = dataset_test[i][0].flatten()

num_classes = 10
mnist_rows = 28
mnist_cols = 28

dim_input = mnist_rows * mnist_cols
fc_dim1 = 100
fc_dim2 = 50
activation = 'pocket_tanh'

fc1 = PktFc(dim_input, fc_dim1, use_dfa=True, activation=activation)
fc2 = PktFc(fc_dim1, fc_dim2, use_dfa=True, activation=activation)
fc_last = PktFc(fc_dim2, num_classes, use_dfa=True, activation=activation)

fc1.set_next_layer(fc2)
fc2.set_next_layer(fc_last)

fc1.forward(mnist_train_images)

UNSIGNED_4BIT_MAX = 15

train_target_mat = PktMat(num_train_samples, num_classes)
num_correct = 0
for r in range(num_train_samples):
    train_target_mat[r][mnist_train_labels[r][0]] = UNSIGNED_4BIT_MAX
    if train_target_mat.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
        num_correct += 1
print(f"Initial training accuracy: {num_correct}, {num_train_samples}, {num_correct / num_train_samples * 100}%")

test_target_mat = PktMat(num_test_samples, num_classes)
num_correct = 0
for r in range(num_test_samples):
    test_target_mat[r][mnist_test_labels[r][0]] = UNSIGNED_4BIT_MAX
    if test_target_mat.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
        num_correct += 1
print(f"Initial testing accuracy: {num_correct / num_test_samples * 100}%")


num_correct = 0
