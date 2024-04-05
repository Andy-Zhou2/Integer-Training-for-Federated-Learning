import numpy as np
from torchvision import datasets, transforms
from pktnn_fc import PktFc
from pktnn_mat import PktMat
from pktnn_consts import UNSIGNED_4BIT_MAX
from pktnn_loss import batch_l2_loss_delta

print('Loading data')
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset_name = 'mnist'  # mnist or fashion_mnist
assert dataset_name in ['mnist', 'fashion_mnist']

if dataset_name == 'mnist':
    dataset_train = datasets.MNIST('../data', train=True,
                                   transform=transform, download=True)
    dataset_test = datasets.MNIST('../data', train=False,
                                  transform=transform, download=True)
elif dataset_name == 'fashion_mnist':
    dataset_train = datasets.FashionMNIST('../data', train=True,
                                          transform=transform, download=True)
    dataset_test = datasets.FashionMNIST('../data', train=False,
                                         transform=transform, download=True)
else:  # should not reach here
    raise ValueError('Invalid dataset name')

num_train_samples = len(dataset_train)
num_test_samples = len(dataset_test)

print('Transforming data')
dataset_train = [((data * 256).numpy().astype(np.int_), answer) for data, answer in dataset_train]
dataset_test = [((data * 256).numpy().astype(np.int_), answer) for data, answer in dataset_test]

num_classes = 10
mnist_rows = 28
mnist_cols = 28

dim_input = mnist_rows * mnist_cols

mnist_train_labels = PktMat(num_train_samples, 1)
mnist_train_images = PktMat(num_train_samples, dim_input)
mnist_test_labels = PktMat(num_test_samples, 1)
mnist_test_images = PktMat(num_test_samples, dim_input)
for i in range(num_train_samples):
    mnist_train_labels[i][0] = dataset_train[i][1]
    mnist_train_images[i] = dataset_train[i][0].flatten()
for i in range(num_test_samples):
    mnist_test_labels[i][0] = dataset_test[i][1]
    mnist_test_images[i] = dataset_test[i][0].flatten()

print('Creating model')
if dataset_name == 'mnist':
    fc_dim1 = 100
    fc_dim2 = 50
    activation = 'pocket_tanh'

    fc1 = PktFc(dim_input, fc_dim1, use_dfa=True, activation=activation)
    fc2 = PktFc(fc_dim1, fc_dim2, use_dfa=True, activation=activation)
    fc_last = PktFc(fc_dim2, num_classes, use_dfa=True, activation=activation)

    fc1.set_next_layer(fc2)
    fc2.set_next_layer(fc_last)
elif dataset_name == 'fashion_mnist':
    fc_dim1 = 200
    fc_dim2 = 100
    fc_dim3 = 50
    activation = 'pocket_tanh'

    fc1 = PktFc(dim_input, fc_dim1, use_dfa=True, activation=activation)
    fc2 = PktFc(fc_dim1, fc_dim2, use_dfa=True, activation=activation)
    fc3 = PktFc(fc_dim2, fc_dim3, use_dfa=True, activation=activation)
    fc_last = PktFc(fc_dim3, num_classes, use_dfa=True, activation=activation)

    fc1.set_next_layer(fc2)
    fc2.set_next_layer(fc3)
    fc3.set_next_layer(fc_last)
else:  # should not reach here
    raise ValueError('Invalid dataset name')

print('Setting target matrix')
train_target_mat = PktMat(num_train_samples, num_classes)
for r in range(num_train_samples):
    train_target_mat[r][mnist_train_labels[r][0]] = UNSIGNED_4BIT_MAX

test_target_mat = PktMat(num_test_samples, num_classes)
for r in range(num_test_samples):
    test_target_mat[r][mnist_test_labels[r][0]] = UNSIGNED_4BIT_MAX

# print('Initial Testing')
# fc1.forward(mnist_train_images)
# num_correct = 0
# for r in range(num_train_samples):
#     if train_target_mat.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
#         num_correct += 1
# print(f"Initial training accuracy: {num_correct}, {num_train_samples}, {num_correct / num_train_samples * 100}%")
#
# fc1.forward(mnist_test_images)
# test_target_mat = PktMat(num_test_samples, num_classes)
# num_correct = 0
# for r in range(num_test_samples):
    # if test_target_mat.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
    #     num_correct += 1
# print(f"Initial testing accuracy: {num_correct / num_test_samples * 100}%")

loss_delta_mat = PktMat()

EPOCH = 100
BATCH_SIZE = 20  # too big could cause overflow

lr_inv = np.int_(1000)

indices = np.arange(num_train_samples)

print('Start training')

for epoch in range(EPOCH):
    print(f'Epoch {epoch}')
    # shuffle indices
    np.random.shuffle(indices)

    if epoch % 10 == 0 and lr_inv < 2 * lr_inv:
        # avoid overflow
        lr_inv *= 2

    sum_loss = 0
    epoch_num_correct = 0
    num_iter = num_train_samples // BATCH_SIZE

    for i in range(num_iter):
        mini_batch_images = PktMat(BATCH_SIZE, dim_input)
        mini_batch_train_targets = PktMat(BATCH_SIZE, num_classes)

        idx_start = i * BATCH_SIZE
        idx_end = idx_start + BATCH_SIZE
        mini_batch_images[0:BATCH_SIZE] = mnist_train_images[indices[idx_start:idx_end]]
        mini_batch_train_targets[0:BATCH_SIZE] = train_target_mat[indices[idx_start:idx_end]]

        fc1.forward(mini_batch_images)

        sum_loss += sum(1 / 2 * np.square(mini_batch_train_targets.mat - fc_last.output.mat).flatten())
        sum_loss_delta = batch_l2_loss_delta(loss_delta_mat, mini_batch_train_targets, fc_last.output)

        for r in range(BATCH_SIZE):
            if mini_batch_train_targets.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
                epoch_num_correct += 1

        # params = [fc1.weight.mat, fc1.bias.mat, fc2.weight.mat, fc2.bias.mat, fc_last.weight.mat, fc_last.bias.mat]
        # print(f'epoch {epoch}: has param sum {sum([np.sum(p) for p in params])}')

        fc_last.backward(loss_delta_mat, lr_inv)

    params = [fc1.weight.mat, fc1.bias.mat, fc2.weight.mat, fc2.bias.mat, fc_last.weight.mat, fc_last.bias.mat]
    print(f'epoch {epoch}: has param sum {sum([np.sum(p) for p in params])}')


    print(
        f'Epoch {epoch}, loss: {sum_loss}, accuracy: {epoch_num_correct / num_train_samples * 100}%')

    # test on test set
    fc1.forward(mnist_test_images)
    num_correct = 0
    for r in range(num_test_samples):
        if test_target_mat.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
            num_correct += 1
    print(f"Testing accuracy: {num_correct / num_test_samples * 100}%")
