# Sports Celebrity Image Classification

This project is an end-to-end *image classification system* that identifies famous sports celebrities using facial features. It combines classical computer vision techniques with machine learning to create a highly accurate classification pipeline.


##  Project Overview

The goal of this project is to classify images of well-known *sports celebrities* (like Virat Kohli, Messi, serena williams, Federer,maria sharapova.) using a machine learning model trained on facial features. The process involves:

- Collecting images
- Detecting facial features
- Extracting meaningful clues using wavelet transforms
- Training a machine learning classifier using GridSearchCV for best performance

##  Data Collection

- Images were collected using [Fatkun Batch Downloader] a Chrome extension that simplifies downloading image batches from the web.
- Images were manually curated to ensure quality and relevance.

## Preprocessing and Feature Extraction

- Face & Eye Detection: Used Haar Cascades from OpenCV (haarcascade_frontalface_default.xml, haarcascade_eye.xml) to detect faces and eyes in the images.
- Wavelet Transform: Applied discrete wavelet transform (DWT) to extract high-frequency facial details (like nose, edges, etc.) that improve classification.
-  Feature Vector: Combined:
  - Raw pixel values (scaled face image)
  - Wavelet-transformed image features into a single feature vector.

## Model Training

- Used GridSearchCV with multiple classifiers (like SVM, Random Forest, Logistic Regression) to find the best-performing model.
- Final model is trained.
