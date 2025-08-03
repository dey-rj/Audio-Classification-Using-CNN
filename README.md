# Audio Classification using CNN

## What is a Mel Spectrogram?
A spectrogram is a visual representation of sound. It plots time on the x-axis, frequency on the y-axis, and the intensity (loudness) of each frequency at a particular time as color.

A Mel spectrogram is a specific type of spectrogram where the frequency axis is transformed to the Mel scale. The Mel scale is a perceptual scale of pitches that mimics how the human ear works. We are much better at distinguishing between low frequencies (e.g., **100 Hz vs 200 Hz**) than we are at distinguishing between high frequencies (e.g., **8000 Hz vs 8100 Hz**). The Mel scale is linear at low frequencies but becomes logarithmic at higher frequencies, effectively grouping high frequencies together while keeping low frequencies spread out.

## How is Mel spectrogram different from a normal audio spectrogram?
The single key difference is the frequency axis.

* Normal Spectrogram: The y-axis is linear (e.g., 0 Hz, 1000 Hz, 2000 Hz, 3000 Hz...). The distance between 1000 and 2000 Hz is the same as the distance between 7000 and 8000 Hz.

* Mel Spectrogram: The y-axis is on the Mel scale. It gives more resolution and space to lower, more perceptually important frequencies and compresses the higher frequencies.


<img src="image\mel_vs_normal.png" alt="Project Logo" width=""/>


## Why can't we use a normal audio spectrogram to classify?
You absolutely can use a normal spectrogram. However, a Mel spectrogram is often preferred for tasks related to human perception (like speech, music, or environmental sounds) for two main reasons:

* Perceptual Relevance: Since the Mel scale is based on human hearing, the features it emphasizes are more likely to be the same features a human would use to identify a sound. This can make the learning task easier for the model. For example, the important, distinguishing characteristics of a dog_bark or siren are concentrated in the lower-to-mid frequency ranges, which the Mel scale highlights.

* Dimensionality Reduction & Efficiency: By compressing the less-important high frequencies into fewer bins, a Mel spectrogram reduces the dimensionality of the input without losing much significant information. This makes the model more computationally efficient and can help it learn more robust features. The provided script uses just 128 Mel bands to represent the entire frequency spectrum.



## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify urban sounds from the UrbanSound8K dataset. The model is built using PyTorch and uses Mel Spectrograms as input features to identify sounds from 10 different classes.

The goal of this project is to build an effective audio classifier. Raw audio waveforms are not ideal for image-based classifiers like CNNs. Therefore, we transform audio signals into a 2D representation called a Mel Spectrogram, which visually represents the spectrum of frequencies as they vary over time. This 2D representation is then used to train a CNN, much like an image classification task.


## Dataset
This project uses the UrbanSound8K dataset, which contains 8732 labeled sound excerpts (each 4 seconds or less).

### 10 Different Audio Class:
| Class Name         | Samples |
|--------------------|---------|
| dog_bark           | 1000    |
| children_playing   | 1000    |
| air_conditioner    | 1000    |
| street_music       | 1000    |
| jackhammer         | 1000    |
| engine_idling      | 1000    |
| drilling           | 1000    |
| siren              | 929     |
| car_horn           | 429     |
| gun_shot           | 374     |



## Methodology

### Feature Extraction: 
Each audio file is loaded and converted into a log-Mel Spectrogram. To ensure uniform input size for the network, spectrograms are padded or truncated to a fixed dimension of 128 x 174.

### Data Preparation:
The extracted features and their corresponding labels are organized into a Pandas DataFrame.

The dataset is split into training (80%) and test (20%) sets.

Labels are then numerically encoded using sklearn.preprocessing.LabelEncoder.

### Model Training:
The AudioCNN model is trained for 50 epochs using the Adam optimizer and Cross-Entropy Loss.

After each epoch, the model is evaluated on the test set, and the model with the best validation accuracy is saved.

### Evaluation:
The best saved model is loaded for final evaluation on the test data.

Performance is visualized through training/validation plots and a confusion matrix.

## Model Architecture
The CNN is designed with four convolutional blocks, followed by a classifier head.

The model input is a Mel Spectrogram of shape (1, 128, 174).

The architecture consists of:

---

### Convolutional Blocks

- **Conv Block 1**  
  `Conv2D (16 filters)` → `ReLU` → `BatchNorm` → `MaxPool` → `Dropout(0.25)`

- **Conv Block 2**  
  `Conv2D (32 filters)` → `ReLU` → `BatchNorm` → `MaxPool` → `Dropout(0.25)`

- **Conv Block 3**  
  `Conv2D (64 filters)` → `ReLU` → `BatchNorm` → `MaxPool` → `Dropout(0.30)`

- **Conv Block 4**  
  `Conv2D (128 filters)` → `ReLU` → `BatchNorm` → `MaxPool` → `Dropout(0.30)`


### Classifier Head

- `AdaptiveAvgPool2D` → `Flatten` → `Linear (128 → 10)`


## Results
---

<img src="image\accuracy loss curve.png" alt="Project Logo" width=""/>
<p align="center"><b>Figure 1:</b> Training & Validation Accuracy/Loss Curve</p>

---
<img src="image\Test confusion matrix.png" alt="Project Logo" width=""/>
<p align="center"><b>Figure 2:</b> Confusion Matrix on Test Data</p>

---
<img src="image\classification report.png" alt="Project Logo" width=""/>
<p align="center"><b>Figure 3:</b> Per-Class Precision, Recall, and F1-Score</p>

### Overall Performance

- **Training Accuracy:** 99.33%  
- **Test Accuracy:** 95.59%
