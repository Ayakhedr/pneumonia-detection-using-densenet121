# Pneumonia Detection from Chest X-rays using DenseNet121

## Project Overview

This project implements an end-to-end **binary image classification pipeline** for detecting pneumonia from chest X-ray images using **Deep Learning and Transfer Learning**.

A pretrained **DenseNet121** model was used as the backbone and adapted to classify chest X-ray images into two classes:

- **NORMAL**
- **PNEUMONIA**

The notebook covers the complete workflow, including dataset exploration, image preprocessing, data loading, transfer learning, model training, validation, evaluation, and final test-set inference.

## Objective

The main objective is to develop and evaluate a deep learning model capable of distinguishing between **Normal** and **Pneumonia** chest X-ray images.

Particular attention was given to model evaluation using multiple metrics, especially **Recall and F1-Score**, rather than relying on accuracy alone.

## Project Workflow

The project follows the following pipeline:

1. Dataset exploration
2. Image and label preparation
3. Class distribution analysis
4. Image visualization
5. Image preprocessing and normalization
6. Custom PyTorch Dataset
7. DataLoader preparation
8. Handling class imbalance
9. Transfer learning using DenseNet121
10. Model architecture adaptation
11. Model training
12. Learning-rate scheduling
13. Early stopping
14. Best model checkpointing
15. Training and validation performance monitoring
16. Classification report
17. Confusion matrix analysis
18. Test-set inference
19. Final model evaluation

## Model

The project uses **DenseNet121** with pretrained ImageNet weights.

Most of the pretrained layers are frozen, while the final classification layer is replaced with a new fully connected layer for binary classification.

The training configuration includes:

- Pretrained DenseNet121
- Transfer Learning
- Cross-Entropy Loss
- Adam Optimizer
- Learning Rate: `1e-4`
- ReduceLROnPlateau Scheduler
- Early Stopping
- Best Model Checkpointing
- CUDA support when available

The best-performing model is saved during training as:

```text
best_Dens121_model.pth
```

## Dataset

The dataset contains chest X-ray images belonging to two classes:

- **NORMAL**
- **PNEUMONIA**

The notebook loads:

- **5,232 training images**
- **624 test images**

The test set contains:

- **234 NORMAL images**
- **390 PNEUMONIA images**

The dataset is loaded from the chest X-ray dataset available in the Kaggle environment used for this project.

## Training

The model was trained for a maximum of **70 epochs**, with **Early Stopping** configured with a patience of 3 epochs.

Training and validation performance were monitored using:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Validation Precision
- Validation Recall
- Validation F1-Score

The best validation performance observed during training reached:

- **Validation Accuracy:** 96.23%
- **Validation Precision:** 99.36%
- **Validation Recall:** 95.70%
- **Validation F1-Score:** 97.50%

Early stopping was triggered after validation performance stopped improving.

## Test Set Results

The final model was evaluated on **624 unseen test images**.

### Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.98 | 0.90 | 0.94 | 234 |
| PNEUMONIA | 0.94 | 0.99 | 0.96 | 390 |
| **Overall Accuracy** | | | **0.95** | **624** |

### Overall Metrics

| Metric | Score |
|---|---:|
| Accuracy | **95%** |
| Macro F1-Score | **0.95** |
| Weighted F1-Score | **0.95** |
| Pneumonia Precision | **0.94** |
| Pneumonia Recall | **0.99** |
| Pneumonia F1-Score | **0.96** |

The high **Pneumonia Recall of 99%** indicates that the model successfully identifies the vast majority of pneumonia cases in the test set.

## Model Evaluation

The model was evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

Probability scores for the **PNEUMONIA** class were also extracted using Softmax, allowing the model's confidence in the pneumonia prediction to be analyzed.

## Key Techniques

### Transfer Learning

Instead of training a CNN completely from scratch, a pretrained DenseNet121 model was used as the feature extractor.

### Fine-Tuning

The pretrained model layers were mostly frozen while the classification head was replaced for the two target classes.

A selected deeper feature layer was also enabled for training.

### Learning Rate Scheduling

`ReduceLROnPlateau` was used to reduce the learning rate when the validation loss stopped improving.

### Early Stopping

Early stopping was used to terminate training when validation loss failed to improve for several consecutive epochs.

### Model Checkpointing

The best model based on validation loss was saved during training and later used for final evaluation.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn

## Project Structure

```text
pneumonia-detection-using-densenet121/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── pneumonia-detection-using-densenet121.ipynb
│
└── models/
    └── best_Dens121_model.pth
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Ayakhedr/pneumonia-detection-using-densenet121.git
cd pneumonia-detection-using-densenet121
```

### 2. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
notebooks/pneumonia-detection-using-densenet121.ipynb
```

Run the notebook cells sequentially to reproduce the project workflow.

> **Note:** The notebook was developed in a Kaggle environment and uses the corresponding Kaggle dataset path. The dataset itself is not included in this repository.

## Key Takeaways

This project demonstrates practical experience with:

- Deep Learning
- Medical Image Classification
- Convolutional Neural Networks
- Transfer Learning
- DenseNet121
- PyTorch
- Torchvision
- Image preprocessing
- Custom Dataset and DataLoader
- Class imbalance handling
- Model fine-tuning
- Learning-rate scheduling
- Early stopping
- Model checkpointing
- Classification metrics
- Confusion matrix analysis
- Test-set inference

##  Disclaimer

This project is intended for **educational and research purposes only**.

The model should not be used as a substitute for professional medical diagnosis or clinical decision-making.
