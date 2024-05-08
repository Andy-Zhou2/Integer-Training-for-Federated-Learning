from dataset.fp_dataset import load_federated_dataset_fp
from dataset.pkt_dataset import load_federated_dataset_pkt


dataset_name = 'mnist'
dataset_dirichlet_alpha = 0.5
num_clients = 100
train_ratio = 1
batch_size = 600

client_datasets_fp, test_dataset_fp = load_federated_dataset_fp(dataset_name, dataset_dirichlet_alpha, num_clients,
                                                                    train_ratio, batch_size, shuffle=False)
client_datasets_pkt, test_dataset_pkt = load_federated_dataset_pkt(dataset_name, dataset_dirichlet_alpha,
                                                                       num_clients, train_ratio)

a = 0
b = 0

import matplotlib.pyplot as plt

fp_image = next(iter(client_datasets_fp[a]['train']))['image']
fp_image = fp_image[b][0]  # shape 28, 28
plt.imshow(fp_image)
plt.show()


pkt_image = client_datasets_pkt[a]['train'][0]
pkt_image = pkt_image[b].reshape(28, 28)
plt.imshow(pkt_image)
plt.show()