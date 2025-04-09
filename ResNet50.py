import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

#For visualization
import matplotlib.pyplot as plt
import numpy as np
import random


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
        
# ----------------------------------------------------------------- ResNet Models -----------------------------------------------------------------  
#       
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

def train(model, train_loader, test_loader, optimizer, criterion, num_epochs=100):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total

        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.5f}, Train Accuracy: {train_accuracy:.5f}%, Test Accuracy: {test_accuracy:.5f}%')

        # Save the best model
        if epoch == 0 or test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved model with accuracy: {best_accuracy:.5f}%')
    
def main():
    # Initialize ResNet50 model, optimizer, and loss function
    model = ResNet50(num_classes=10)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize the cifar-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Visualize the the 10 classes in the dataset with 5 images each
    classes = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    class_indices = {class_name: [] for class_name in classes}
    for idx, (image, label) in enumerate(train_dataset):
        class_name = classes[label]
        if len(class_indices[class_name]) < 5:
            class_indices[class_name].append(idx)
    # Randomly select 5 images from each class
    selected_indices = []
    for class_name, indices in class_indices.items():
        selected_indices.extend(random.sample(indices, 5))
    # Plot the images
    fig, axes = plt.subplots(5, 10, figsize=(15, 7))
    for i, idx in enumerate(selected_indices):
        image, label = train_dataset[idx]
        ax = axes[i // 10, i % 10]
        ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        ax.set_title(classes[label])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Train and validate the model and print loss, train accuracy, and test accuracy
    # Training loop
    num_epochs = 200
    train(model, train_loader, test_loader, optimizer, criterion, num_epochs)

    #Load the best model
    model = ResNet50(num_classes=10)
    model.load_state_dict(torch.load('best_model.pth'))

    # Test the model by classifying 10 random images from the test set
    random_indices = random.sample(range(len(test_dataset)), 10)
    images = []
    labels = []
    for idx in random_indices:
        image, label = test_dataset[idx]
        images.append(image)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    # Print the predicted and actual labels along with images
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for i, idx in enumerate(random_indices):
        image, label = test_dataset[idx]
        ax = axes[i // 5, i % 5]
        ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        ax.set_title(f'Pred: {predicted[i].item()}, Actual: {label}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
        