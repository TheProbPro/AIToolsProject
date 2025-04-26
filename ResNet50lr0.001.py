import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm
import tqdm.auto as tqdm

import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score
)
from sklearn.preprocessing import label_binarize

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

# ----------------------------------------------------------------- Training and Testing -----------------------------------------------------------------

def load_CIFAR10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
"""
def train(model, train_loader, test_loader, optimizer, criterion, num_epochs=100):
    best_accuracy = 0.0
    for epoch in tqdm.tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
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
"""
def train(model, train_loader, test_loader, optimizer, criterion, num_epochs=100):
    best_accuracy = 0.0

    # Lists to store metrics for plotting
    epoch_nums = list(range(1, num_epochs + 1))
    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in tqdm.tqdm(range(num_epochs), desc="Epoch"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # compute epoch metrics
        epoch_loss = running_loss / total
        epoch_train_acc = 100.0 * correct / total

        # Validation loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_test_acc = 100.0 * val_correct / val_total

        # Save metrics
        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(epoch_test_acc)

        # print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {epoch_train_acc:.2f}% | "
                  f"Test Acc: {epoch_test_acc:.2f}%")

        # Save the best model
        if epoch == 0 or epoch_test_acc > best_accuracy:
            best_accuracy = epoch_test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"→ Saved new best model (Test Acc: {best_accuracy:.2f}%)")

    # After training, plot each metric in its own figure:

    # 1) Training Loss
    plt.figure(figsize=(6,4))
    plt.plot(epoch_nums, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Training Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epoch_nums, train_accs, label='Train Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Test Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epoch_nums, test_accs, label='Test Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_loader):
    """
    Runs the model on the test set and computes:
      - Confusion matrix (plotted)
      - Per-class accuracy, sensitivity, specificity, precision, F1
      - Overall accuracy
      - Classification report
      - ROC curves & AUC for each class
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probas, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probas.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # Get class names from the dataset
    class_names = test_loader.dataset.classes

    # 1) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # 2) Compute per-class metrics
    total = cm.sum()
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = total - (tp + fp + fn)

    sensitivity = tp / (tp + fn)           # recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)

    print("Per-class metrics:")
    for i, cls in enumerate(class_names):
        acc_i = (tp[i] + tn[i]) / total
        print(
            f"  {cls:10s}  Acc: {acc_i:.4f}  "
            f"Sens: {sensitivity[i]:.4f}  Spec: {specificity[i]:.4f}  "
            f"Prec: {precision[i]:.4f}  F1: {f1[i]:.4f}"
        )

    # 3) Overall accuracy & full report
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 4) ROC curves & AUC
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    plt.figure(figsize=(8, 8))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Initialize ResNet50 model, optimizer, and loss function
    model = ResNet50(num_classes=10)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Model initialized.")

    # Initialize the cifar-10 dataset
    train_loader, test_loader = load_CIFAR10(batch_size=64)
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    print("CIFAR-10 dataset loaded.")

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
    print("Starting training...")
    num_epochs = 200
    train(model, train_loader, test_loader, optimizer, criterion, num_epochs)
    print("Training completed.")

    #Load the best model
    print("Loading the best model...")
    model = ResNet50(num_classes=10)
    model.load_state_dict(torch.load('best_model.pth'))
    model.cuda()

    # Evaluate the model on the test set
    print("Evaluating the model...")
    evaluate_model(model, test_loader)
    print("Evaluation completed.")

    # Test the model by classifying 10 random images from the test set
    print("Testing the model on random images...")
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
    print("Testing completed.")
        

main()