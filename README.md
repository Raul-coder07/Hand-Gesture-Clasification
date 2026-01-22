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
