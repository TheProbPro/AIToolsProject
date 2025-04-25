# CIFAR-10 CNN Classifier - Comprehensive Evaluation

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using PyTorch. It includes a detailed training process, rich evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC), and visualizations like confusion matrix and training curves.

---

## Project Structure

```bash
├── cifar10_classifier.py       # Main Python script (Jupyter Notebook compatible)
├── data/                       # CIFAR-10 data (auto-downloaded by torchvision)
├── outputs/                    # Optional: store plots or results here
└── README.md                   # Project documentation
```

---

## Dataset

**CIFAR-10** is a dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

- **Training samples**: 50,000
- **Testing samples**: 10,000
- **Classes**: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

---

## Dependencies

Ensure you have the following Python packages installed:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pandas tqdm
```

Optional: use a virtual environment or `conda`.

---

## How to Run

1. Clone this repository or copy the Python script / notebook.
2. Run the script using:

```bash
python cifar10_classifier.py
```

> Alternatively, use Jupyter Notebook or VS Code for step-by-step execution and visualization.

---

## Model Architecture

The CNN consists of:

- 3 Convolutional blocks:
  - Conv2D + BatchNorm + ReLU + MaxPooling
- 1 Fully connected (Linear) layer
- Dropout for regularization
- Final output layer for classification

Input shape: `3 x 32 x 32`  
Output: 10-class logits

---

## Training Details

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (lr = 0.001)
- **Epochs**: 10
- **Batch Size**: 64
- **Device**: CUDA (if available)

During training, the script logs:

- Epoch-wise loss and accuracy
- Accuracy trend and loss trend plots

---

## Evaluation Metrics

After training, the model is evaluated using:

- Overall Accuracy
- Per-class Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve + AUC for each class
- Classification Report (text output)

---

## Visualizations

The script generates the following plots:

- **Training Loss / Accuracy**
- **Confusion Matrix**
- **Accuracy per Class**
- **Precision, Recall, F1-score Bar Charts**
- **Support (sample count per class)**
- **ROC Curves with AUC**

These visualizations give deep insights into performance across all CIFAR-10 classes.

---

## Notes

- This script uses random horizontal flips and random cropping for data augmentation.
- You can improve performance by training for more epochs, using learning rate scheduling, or applying regularization.

---

## To Do (Optional Enhancements)

- Implement test-time augmentation (TTA)
- Add learning rate scheduler (e.g., ReduceLROnPlateau)
- Use ResNet or other architectures for comparison
- Save model checkpoints and reload for inference

---

## Author

Made by [Zichen Wang]  
Feel free to fork or improve!

---

## License

This project is open source and available under the [MIT License](LICENSE).
