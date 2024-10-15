# Advance-Image-Processing
Assignments of AIP course


# Image Segmentation and Denoising Techniques

This repository contains implementations of various image segmentation and denoising techniques using deep learning and traditional machine learning methods.

## Table of Contents
1. [NCut Segmentation](#ncut-segmentation)
2. [K-Means Segmentation](#k-means-segmentation)
3. [PCA-SIFT](#pca-sift)
4. [ResNet for Image Segmentation](#resnet-for-image-segmentation)
5. [YOLO v5 Fine Tuning](#yolo-v5-fine-tuning)
6. [Image Denoising](#image-denoising)

---

## NCut Segmentation
Normalized Cut (NCut) is a graph-based image segmentation method used to partition an image into meaningful regions. This approach utilizes the eigenvalues of a similarity matrix to segment the image by minimizing the normalized cut between regions.

### Features:
- Graph-based segmentation
- Eigenvalue decomposition

## K-Means Segmentation
Definition: K-Means is a clustering algorithm that partitions the image into K clusters by minimizing the within-cluster variance. In image segmentation, pixels with similar color or intensity are grouped together based on their proximity in feature space.

## Features:
Unsupervised clustering of pixels
Easy and computationally efficient
Clusters pixels based on color or intensity

## PCA-SIFT
Definition: Principal Component Analysis (PCA) applied to Scale-Invariant Feature Transform (SIFT) is a technique used to reduce the dimensionality of the SIFT feature descriptors. This allows for efficient storage and faster matching while retaining the most important aspects of the features.

## Features:
Dimensionality reduction of SIFT descriptors
Efficient feature matching
Helps in image retrieval and object recognition tasks

## ResNet for Image Segmentation
Definition: ResNet (Residual Networks) is a deep learning model that uses residual connections to enable the training of very deep neural networks. In this project, a pre-trained ResNet model is fine-tuned for image segmentation, which involves classifying each pixel of the image.

## Features:
Pre-trained deep convolutional network
Transfer learning for custom image segmentation tasks
High accuracy due to deep feature extraction

## YOLO v5 Fine Tuning
Definition: You Only Look Once (YOLO) is a real-time object detection system. YOLO v5 is a widely-used variant that performs object detection by dividing the image into a grid and predicting bounding boxes and class probabilities simultaneously. This implementation fine-tunes a pre-trained YOLO v5 model on a custom dataset to improve detection performance in a specific domain.

## Features:
Real-time object detection
Fine-tuning for domain-specific improvements
Efficient and accurate bounding box predictions


## Image Denoising
Definition: Image denoising is a technique used to remove noise from an image while preserving important details. This can be achieved using traditional filtering techniques (such as Gaussian and median filtering) or deep learning methods, such as denoising autoencoders.

## Features:
Reduces noise in images, improving quality
Supports traditional and deep learning-based methods
Can be applied to various types of noise (e.g., Gaussian, Salt-and-Pepper)
