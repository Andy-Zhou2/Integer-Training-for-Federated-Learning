import numpy as np
from torchvision import datasets, transforms
from pktnn_fc import PktFc
from pktnn_mat import PktMat
from pktnn_consts import UNSIGNED_4BIT_MAX
from pktnn_loss import batch_l2_loss_delta
from state import save_state, load_state
import os

print('Loading data')

dataset_name = 'fashion_mnist'  # mnist or fashion_mnist
assert dataset_name in ['mnist', 'fashion_mnist']

if dataset_name == 'mnist':
    dataset_train = datasets.MNIST('../data', train=True,
                                   download=True)
    dataset_test = datasets.MNIST('../data', train=False,
                                  download=True)
elif dataset_name == 'fashion_mnist':
    dataset_train = datasets.FashionMNIST('../data', train=True,
                                          download=True)
    dataset_test = datasets.FashionMNIST('../data', train=False,
                                         download=True)
else:  # should not reach here
    raise ValueError('Invalid dataset name')

# create folder
weight_folder = f'../data/weights/{dataset_name}'
os.makedirs(weight_folder, exist_ok=True)

num_train_samples = len(dataset_train)
num_test_samples = len(dataset_test)

print('Transforming data')
num_classes = 10
mnist_rows = 28
mnist_cols = 28

dim_input = mnist_rows * mnist_cols

mnist_train_labels = PktMat(num_train_samples, 1)
mnist_train_images = PktMat(num_train_samples, dim_input)
mnist_test_labels = PktMat(num_test_samples, 1)
mnist_test_images = PktMat(num_test_samples, dim_input)

mnist_train_labels.mat = dataset_train.targets.numpy().reshape(-1, 1)
mnist_train_images.mat = dataset_train.data.numpy().reshape(-1, dim_input)
mnist_test_labels.mat = dataset_test.targets.numpy().reshape(-1, 1)
mnist_test_images.mat = dataset_test.data.numpy().reshape(-1, dim_input)

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

    fc_list = [fc1, fc2, fc_last]
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

    fc_list = [fc1, fc2, fc3, fc_last]
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



EPOCH = 100
BATCH_SIZE = 20  # too big could cause overflow

lr_inv = np.int_(1000)

# load state
START_EPOCH = 1
# load_state(fc_list, os.path.join(weight_folder, f'epoch_{START_EPOCH-1}.npz'))
# lr_inv = np.int_(2000)

indices = np.arange(num_train_samples)

print('Start training')

for epoch in range(START_EPOCH, EPOCH + 1):
    print(f'Epoch {epoch}')
    # shuffle indices
    # TODO: shuffle indices
    # np.random.shuffle(indices)

    if epoch > 10:
        print('DEBUG')

    if epoch % 10 == 0 and lr_inv < 2 * lr_inv:
        # avoid overflow
        lr_inv *= 2

    sum_loss = 0
    epoch_num_correct = 0
    num_iter = num_train_samples // BATCH_SIZE

    for i in range(num_iter):
        # print('\n')
        # print('iter:', i)

        mini_batch_images = PktMat(BATCH_SIZE, dim_input)
        mini_batch_train_targets = PktMat(BATCH_SIZE, num_classes)

        idx_start = i * BATCH_SIZE
        idx_end = idx_start + BATCH_SIZE
        mini_batch_images[0:BATCH_SIZE] = mnist_train_images[indices[idx_start:idx_end]]
        mini_batch_train_targets[0:BATCH_SIZE] = train_target_mat[indices[idx_start:idx_end]]

        # print('mini_batch_images:', mini_batch_images.sum())
        fc1.forward(mini_batch_images)

        # if i == 0:
        #     # print output
        #     print(f'target:')
        #     mini_batch_train_targets.print()
        #     print(f'batch 1 output:')
        #     fc_last.output.print()

        sum_loss += sum(np.square(mini_batch_train_targets.mat - fc_last.output.mat).flatten() // 2)
        loss_delta_mat = batch_l2_loss_delta(mini_batch_train_targets, fc_last.output)

        for r in range(BATCH_SIZE):
            if mini_batch_train_targets.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
                epoch_num_correct += 1

        # params = [fc1.weight.mat, fc1.bias.mat, fc2.weight.mat, fc2.bias.mat, fc_last.weight.mat, fc_last.bias.mat]
        # print(f'epoch {epoch}: has param sum {sum([np.sum(p) for p in params])}')

        fc_last.backward(loss_delta_mat, lr_inv)


    params_sum = sum([fc.weight.sum() + fc.bias.sum() for fc in fc_list])
    print(f'epoch {epoch} iter {None}: has param sum {params_sum}')
    for fc in fc_list:
        print(f'fc hash: {fc.weight.hash()} {fc.bias.hash()}')

    # save state
    save_state(fc_list, os.path.join(weight_folder, f'epoch_{epoch}.npz'))

    print(
        f'Epoch {epoch}, loss: {sum_loss}, accuracy: {epoch_num_correct / num_train_samples * 100}%')

    # test on test set
    fc1.forward(mnist_test_images)
    num_correct = 0
    for r in range(num_test_samples):
        if test_target_mat.get_max_index_in_row(r) == fc_last.output.get_max_index_in_row(r):
            num_correct += 1
    print(f"Testing accuracy: {num_correct / num_test_samples * 100}%")
