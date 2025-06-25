# Diffusion Models for Time Series Generation

This repository contains a Python-based implementation of a diffusion model tailored for generating synthetic time series data. The core of the project is built in a Jupyter Notebook (`timeserie_diffusion.ipynb`) and leverages PyTorch for model building and training.

## Introduction

Time series data is ubiquitous in many fields, from finance to weather forecasting. Generating realistic synthetic time series data can be valuable for data augmentation, imputation, and creating anonymized datasets.

This project explores the use of Denoising Diffusion Probabilistic Models (DDPMs) to learn the underlying distribution of a given time series dataset and generate new, unseen samples that share the same statistical properties. The model learns to progressively add noise to the data in a "forward process" and then, more importantly, learns to reverse this process, starting from pure noise to generate a clean time series sample.

## How it Works: Diffusion for Time Series

The core idea is based on a two-step process:

1.  **Forward Process (Noise Addition)**: We take a real time series sample and gradually add Gaussian noise over a series of `T` timesteps. As `t` approaches `T`, the data sample becomes indistinguishable from pure noise. This process is mathematically defined and does not involve any learnable parameters.

2.  **Reverse Process (Denoising)**: This is where the learning happens. We train a neural network (in this case, a U-Net-like architecture) to predict the noise that was added at each timestep `t`. By iteratively subtracting the predicted noise from a random noise vector, the model can reconstruct a new, clean time series sample.

The training objective is to optimize the neural network to accurately predict the added noise, typically using a Mean Squared Error (MSE) loss between the actual noise and the predicted noise.

## Repository Structure

```bash
.
├── data
│   └── pollution.csv
├── extra
│   ├── pollutionDatasetDemo.ipynb
│   ├── timeserie_diffusion_colab.ipynb
│   └── visualisations.ipynb
├── model
│   ├── diffusion.py
│   ├── noise_scheduler.py
│   ├── __pycache__
│   │   ├── diffusion.cpython-312.pyc
│   │   ├── noise_scheduler.cpython-312.pyc
│   │   ├── train.cpython-312.pyc
│   │   └── unet.cpython-312.pyc
│   ├── train.py
│   └── unet.py
├── notes.md
├── __pycache__
├── README.md
├── samples plots
│   ├── pollution.png
│   ├── synt_long.png
│   └── sytn_short.png
├── timeserie_diffusion.ipynb
└── util
    ├── evaluation.py
    ├── pollutionDataset.py
    ├── syntheticDataset.py
    └── vizUtil.py
```


## Model Architecture

The core of our generative model is the denoising network, a sophisticated neural architecture tasked with a single, crucial objective: precisely estimate the noise present in a corrupted time series sample at any given step of the reverse diffusion process. For this, we employ a variant of the powerful U-Net architecture.

The U-Net is exceptionally well-suited for generative tasks like this because it excels at processing information at multiple resolutions simultaneously. Its structure can be broken down into three key components:
1. **The Encoder (Down-sampling Path)**.  The encoder acts like a feature extractor. It takes the noisy time series as input and progressively down-samples it using a series of convolutional layers. At each step, the time series becomes shorter, but its feature representation becomes deeper. This process forces the network to capture the broader, low-frequency patterns of the time series—such as its overall trend, seasonality, and general shape.
2. **The Decoder (Up-sampling Path)**. The decoder's job is to reconstruct the estimated noise from the compressed, high-level features created by the encoder. It progressively up-samples the feature maps, moving from a coarse representation back to the original time series length.
3. **Skip Connections**. This is the U-Net's defining feature. Skip connections create a direct link between the encoder and decoder at corresponding resolutions. These connections allow the decoder to access not just the high-level abstract features but also the fine-grained, high-frequency details (like sharp peaks or subtle textures) that were captured by the early encoder layers. For time series, this is critical for generating realistic, non-blurry results and preserving the subtle nuances of the original data.
