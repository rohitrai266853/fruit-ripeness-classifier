# Fruit Ripeness Classifier

A Convolutional Neural Network (CNN) based image classification project that classifies fruits into three categories:

* UnRipe
* Ripe
* OverRipe

## Overview

This project uses TensorFlow and Keras to train a deep learning model on fruit images. The model learns visual features associated with different ripeness stages and predicts the ripeness class of a given fruit image.

## Dataset

The dataset contains images organized into three folders:

```text
OverRipe/
Ripe/
UnRipe/
```

Dataset split:

* Training Images: 781
* Validation Images: 194

## Technologies Used

* Python 3.11
* TensorFlow
* Keras
* NumPy
* Matplotlib
* OpenCV

## Model Architecture

* Conv2D (32 filters)
* MaxPooling2D
* Conv2D (64 filters)
* MaxPooling2D
* Flatten
* Dense (128 neurons)
* Dense (3 output classes, Softmax)

## Results

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 97.57% |
| Validation Accuracy | 94.33% |

## Project Structure

```text
fruit-ripeness-classifier/
│
├── OverRipe/
├── Ripe/
├── UnRipe/
├── train.py
├── fruit_model.keras
└── README.md
```

## Installation

```bash
pip install tensorflow opencv-python matplotlib numpy
```

## Run

```bash
python train.py
```

## Output

The trained model is saved as:

```text
fruit_model.keras
```

## Future Improvements

* Web interface using Streamlit or Flask
* Support for custom image uploads
* Real-time ripeness detection
* Mobile deployment
* Support for additional fruit types

## Author

Rohit Rai
