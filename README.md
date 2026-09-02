# Unsupervised Speech Anomaly Detection

## Overview

This project investigates unsupervised speech anomaly detection using
pretrained speech representations and an autoencoder.

The main idea is:

Raw Speech
    ↓
Audio Preprocessing
    ↓
Pretrained Speech Encoder
    ↓
Speech Embeddings
    ↓
Autoencoder
    ↓
Reconstruction Error
    ↓
Anomaly Score
    ↓
Evaluation

## Motivation

The project explores whether pretrained speech representations can
capture useful patterns for distinguishing atypical speech samples
without requiring a fully supervised classifier.

## Research Questions

1. Can pretrained speech representations provide useful features
   for unsupervised anomaly detection?

2. Can an autoencoder learn a representation of normal speech?

3. Does reconstruction error provide a useful anomaly score?

4. How does the choice of pretrained representation affect detection
   performance?

## Technologies

- Python
- PyTorch
- Transformers
- Wav2Vec2
- Librosa
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Project Status

🚧 Work in progress.