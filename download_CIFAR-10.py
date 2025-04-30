import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    # CIFAR-10 images are 32x32; MobileNet needs 224x224
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, num_workers=4)


# Load unnormalized images to visualize them in true color
transform_vis = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

vis_dataset = datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_vis)
vis_loader = DataLoader(vis_dataset, batch_size=8, shuffle=True)

# Class labels for CIFAR-10
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
classes = vis_dataset.classes

# Grab a batch
images, labels = next(iter(vis_loader))

# Convert from tensors to numpy for plotting


def imshow(img_tensor):
    img = img_tensor.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis('off')


# Plot images in a row
plt.figure(figsize=(12, 4))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    imshow(images[i])
    plt.title(classes[labels[i]])
plt.tight_layout()
plt.show()
