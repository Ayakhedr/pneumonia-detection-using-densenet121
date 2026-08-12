# Pneumonia Detection from Chest X-rays using DenseNet121

## Project Overview

This project implements an end-to-end **binary image classification pipeline** for detecting pneumonia from chest X-ray images using **Deep Learning and Transfer Learning**.

A pretrained **DenseNet121** model was used as the backbone and adapted to classify chest X-ray images into two classes:

- **NORMAL**
- **PNEUMONIA**

The notebook covers the complete workflow, including dataset exploration, image preprocessing, data preparation, transfer learning, model training, validation, evaluation, and final test-set inference using PyTorch.

---

# Objective

The main objective of this project is to develop a deep learning model capable of distinguishing between **Normal** and **Pneumonia** chest X-ray images.

The model was evaluated using multiple classification metrics, including:

- Precision
- Recall
- F1-Score
- Confusion Matrix

rather than relying only on accuracy.

---

# Project Workflow

The project follows a complete computer vision pipeline:

1. Dataset exploration
2. Image and label preparation
3. Class distribution analysis
4. Image visualization
5. Image preprocessing and normalization
6. Custom PyTorch Dataset implementation
7. DataLoader preparation
8. Handling class imbalance
9. Transfer learning using DenseNet121
10. Model architecture modification
11. Model training
12. Learning-rate scheduling
13. Early stopping
14. Best model checkpointing
15. Training and validation performance monitoring
16. Classification report
17. Confusion matrix analysis
18. Final test-set inference
19. Model evaluation

---

# Dataset

The dataset contains chest X-ray images belonging to two classes:

- **NORMAL**
- **PNEUMONIA**

The notebook uses:

- **5,232 training images**
- **624 test images**

The test set contains:

- **234 NORMAL images**
- **390 PNEUMONIA images**

The dataset was obtained from a Kaggle chest X-ray dataset.

The dataset itself is not included in this repository.

---

# Data Leakage Prevention

Medical imaging datasets may contain multiple images from the same patient.

To ensure a realistic evaluation and prevent data leakage, patient-level splitting was applied during dataset preparation.

This prevents images from the same patient from appearing in both training and validation sets.

As a result, validation performance better represents how the model performs on unseen patients.

---

# Model

The project uses:

## DenseNet121

DenseNet121 was selected as the CNN backbone using pretrained ImageNet weights.

The pretrained network was adapted for binary classification by replacing the original classifier layer with a new fully connected layer.

The training strategy included:

- Transfer Learning
- Partial fine-tuning
- Cross-Entropy Loss
- Adam Optimizer
- Learning Rate: `1e-4`
- ReduceLROnPlateau Scheduler
- Early Stopping
- Best Model Checkpointing
- CUDA acceleration when available

Initially, most pretrained layers were frozen. Selected deeper layers were enabled for training to allow the model to adapt better to chest X-ray features.

---

# Training

The model was trained for a maximum of **70 epochs**.

Early stopping was applied with a patience of 3 epochs to reduce overfitting and stop training when validation performance stopped improving.

Training performance was monitored using:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Validation Precision
- Validation Recall
- Validation F1-Score

The best-performing checkpoint based on validation performance was saved and later used for final evaluation.

---

# Best Validation Performance

- **Validation Accuracy:** 96.23%
- **Validation Precision:** 99.36%
- **Validation Recall:** 95.70%
- **Validation F1-Score:** 97.50%

---

# Test Set Evaluation

The final model was evaluated on **624 unseen test images**.

The test set was not used during training or model selection.

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| NORMAL | 0.98 | 0.90 | 0.94 | 234 |
| PNEUMONIA | 0.94 | 0.99 | 0.96 | 390 |

---

## Overall Metrics

| Metric | Score |
|--------|-------|
| Accuracy | **95%** |
| Macro F1-Score | **0.95** |
| Weighted F1-Score | **0.95** |
| Pneumonia Precision | **0.94** |
| Pneumonia Recall | **0.99** |
| Pneumonia F1-Score | **0.96** |

The model achieved a **99% recall for the PNEUMONIA class**, meaning it successfully detected the majority of pneumonia cases in the test set.

---

# Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

Prediction probabilities were extracted using Softmax to analyze model confidence.

---

# Key Techniques

## Transfer Learning

Instead of training a CNN from scratch, DenseNet121 pretrained on ImageNet was used as a feature extractor.

This helps achieve better performance with medical image datasets.

---

## Fine-Tuning

The pretrained model was adapted by:

- Replacing the final classification layer
- Freezing most pretrained layers
- Training selected deeper layers to improve adaptation to chest X-ray features

---

## Learning Rate Scheduling

`ReduceLROnPlateau` was used to automatically reduce the learning rate when validation loss stopped improving.

---

## Early Stopping

Early stopping was applied to prevent unnecessary training and reduce overfitting.

---

## Model Checkpointing

The best model checkpoint was saved during training and used for final test evaluation.

---

# Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- OpenCV
- Pillow
- Matplotlib
- Scikit-learn

---

# Project Structure

```
pneumonia-detection-using-densenet121/

│
├── README.md
├── requirements.txt
│
└── notebooks/
    └── pneumonia-detection-using-densenet121.ipynb
```

---

# How to Run

## 1. Clone the repository

```bash
git clone https://github.com/Ayakhedr/pneumonia-detection-using-densenet121.git

cd pneumonia-detection-using-densenet121
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run the notebook

Open:

```
notebooks/pneumonia-detection-using-densenet121.ipynb
```

Run the notebook cells sequentially.

---

# Model Weights

The trained DenseNet121 model checkpoint is not included in this repository because the file size exceeds GitHub's standard upload limitations.

The complete training pipeline is available in the notebook, and the model can be reproduced by running the training process.

---

# Limitations

- The dataset may not represent all real-world clinical environments.
- Model performance may vary when applied to external hospital datasets.
- The model is developed for research and educational purposes and should not be considered a clinical diagnostic tool.

---

# Key Takeaways

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
- Test-set evaluation

---

# Disclaimer

This project is intended for **educational and research purposes only**.

It is not intended to replace professional medical diagnosis or clinical decision-making.
