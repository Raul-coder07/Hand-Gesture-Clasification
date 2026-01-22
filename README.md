# EMG Hand Gesture Classification

This repository contains deep learning models for hand gesture classification using EMG signals.

## Datasets

Three datasets are used:

- Dataset 1: BTS sensors, Subject 2 (hand amputation), 6 channels
- Dataset 2: MyoWare 2.0, Subject 2 (hand amputation), 4 channels
- Dataset 3: MyoWare 2.0, Subject 1 (healthy), 4 channels

Sampling frequency: 1 kHz.

## Gestures

The following gestures are included:

- Index finger with thumb
- Middle finger with thumb
- Ring finger with thumb
- Little finger with thumb
- Fist
- Open hand
- Rest

Note: The datset 1 does not include the open-hand gesture.

## Models

The repository includes CNN-based models using Conv2D layers and a model using dual-input with RMS envelope.

## Quantization

All trained models were converted to TensorFlow Lite (TFLite) to reduce inference time.

## Objective

The objective is to evaluate EMG-based gesture classification using medical-grade and low-cost sensors.

## Models Code Availability

This repository includes the complete code used to train and evaluate **Model 1** and **Model 2**.

- The folder **Model 1/** contains all the scripts related to Model 1.
  - The code is provided using **Dataset 2** as an example.
  - Although three different datasets were used in the experiments, the same model architecture and training pipeline were applied to all of them.

- The folder **Model 2/** follows the same structure and logic.
  - It contains the full implementation of Model 2.
  - The scripts are also configured using **Dataset 2** as a representative example.

Please note that **the datasets are not included in this repository**, as they are private and cannot be publicly shared. However, the code is fully functional and can be adapted to other datasets with the same structure.

