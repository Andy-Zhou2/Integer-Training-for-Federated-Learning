from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Define the index of the image you want to display
index = 73

# Load the MNIST test dataset
test_dataset = MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())

# Retrieve the image and label by index
image, label = test_dataset[index]

# Display the image and its label
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()
