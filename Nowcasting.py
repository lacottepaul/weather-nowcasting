#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nowcasting Radar Forecasting Script
Author: <Lacotte Paul>
Date: 2025-07-02
Description: Forecast next radar images based on past reflectivity maps using both
             a persistence model and a deep learning model (ConvLSTM/U-Net).
"""


import os
import time
import argparse
import logging


import numpy as np
import pandas as pd
import scipy
import xarray as xr
import netCDF4


import matplotlib
matplotlib.use('Agg')  # backend non interactif pour sauvegarde d’images
import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import DataLoader, Dataset


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -----------------------------------
# 1. Data Handling
# -----------------------------------

def load_radar_dataset(path):
    """
    Load the dataset from a netcdf file

    Returns:
        ds: xarray.Dataset
        ref: xarray.DataArray main variable #we only have reflectivity in our file here
        data: np.ndarray 
    """
    ds = xr.open_dataset(path)
    var = list(ds.data_vars)[0]  #name of variable
    ref = ds[var]
    data = ref.values
    return ds, ref, data

def print_dataset_summary(ref):
    """
    print the summary of ds
    """
    print("Min:", float(ref.min()))
    print("Max:", float(ref.max()))
    print("Mean:", float(ref.mean()))
    print("Std Dev:", float(ref.std()))
    print("Nombre de NaNs:", np.isnan(ref.values).sum())
    print("Timestamps available (first 5):")
    print(ref.time.values[:5])

def plot_first_images(ref):
    """
    Plot the first images to get an idea
    """
    plt.imshow(ref[0], cmap='viridis')
    plt.title('Image at t')
    plt.colorbar()
    plt.figure()
    plt.imshow(ref[1], cmap='viridis')
    plt.title('Image at t+1')
    plt.colorbar()
    plt.show()

def create_unet_sequences(data, n_input=3, n_output=1):
    """
    Create the format for the inputs and targets for the Unet.

    Args:
        data: np.ndarray (shape: time, H, W)
        n_input: nbr of inputs
        n_output: nbr of outputs

    Returns:
        X: np.ndarray (shape: N, H, W, n_input)
        Y: np.ndarray (shape: N, H, W, 1)
    """
    X, Y = [], []
    for i in range(len(data) - n_input - n_output + 1):
        x_seq = np.stack([data[i + j] for j in range(n_input)], axis=-1)  # (H, W, n_input)
        y_seq = data[i + n_input]  # (H, W)
        X.append(x_seq)
        Y.append(y_seq)
    return np.array(X), np.array(Y)[..., np.newaxis]  # Add a channel to Y


# -----------------------------------
# 2. Baseline Model (Persistence)
# -----------------------------------

def persistence_model(X):
    """
    Persistence model: prediction = last input frame repeated.
    """
    last_frame = X[..., -1]          # (N, H, W)
    return last_frame[..., np.newaxis]


# -----------------------------------
# 3. Deep Learning Model (Unet)
# -----------------------------------

def build_simple_unet(input_shape):
    """
    Builds a very simple U-Net with 2 levels of downsampling and upsampling.

    Args:
        input_shape (tuple): shape of the input (height, width, channels)

    Returns:
        model (tf.keras.Model): compiled U-Net model
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(b)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)

    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c4)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='linear')(c4)

    model = models.Model(inputs, outputs)
    return model


# -----------------------------------
# 4. Evaluation & Visualization
# -----------------------------------

def evaluate_model(y_true, y_pred):
    """Evaluate prediction using RMSE."""
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    logging.info(f"RMSE: {rmse:.4f}")
    return rmse


def plot_prediction(input_seq, true_seq, pred_seq, sample_idx=0, pred_step=0, model_name="model"):
    Input = input_seq[sample_idx, :, :, -1]
    Target = true_seq[sample_idx, :, :, 0]
    Prediction = pred_seq[sample_idx, :, :, 0]

    # Plot Input
    plt.figure()
    plt.imshow(Input, cmap='viridis')
    plt.title('Image at t (Last Input Frame)')
    plt.colorbar()
    plt.savefig(f"{model_name}_input_frame.png")
    plt.close()

    # Plot Target
    plt.figure()
    plt.imshow(Target, cmap='viridis')
    plt.title('True Image at t+1')
    plt.colorbar()
    plt.savefig(f"{model_name}_true_frame.png")
    plt.close()

    # Plot Prediction
    plt.figure()
    plt.imshow(Prediction, cmap='viridis')
    plt.title('Predicted Image at t+1')
    plt.colorbar()
    plt.savefig(f"{model_name}_pred_frame.png")
    plt.close()

    # Plot Error
    plt.figure()
    plt.imshow(np.abs(Target - Prediction), cmap='hot')
    plt.title('Error (Abs Difference)')
    plt.colorbar()
    plt.savefig(f"{model_name}_error_frame.png")
    plt.close()


# -----------------------------------
# 5. Main Execution Logic
# -----------------------------------

def main(args):
    # Load and preprocess data
    ds, ref, data = load_radar_dataset(args.data_path)
    print_dataset_summary(ref)
    plot_first_images(ref)
    
    X, y = create_unet_sequences(data)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.model == "baseline":
        logging.info("Running baseline (persistence) model...")
        y_pred = persistence_model(X_test)
        evaluate_model(y_test, y_pred)
        plot_prediction(X_test, y_test, y_pred,model_name=args.model)

    elif args.model == "unet":
        input_shape = X_train.shape[1:]  # (H, W, n_input)
        model = build_simple_unet(input_shape)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        logging.info("Training U-Net model...")
        model.fit(X_train, y_train, validation_split=0.1, epochs=args.epochs, batch_size=5)

        y_pred = model.predict(X_test)
        evaluate_model(y_test, y_pred)
        plot_prediction(X_test, y_test, y_pred,model_name=args.model)
        model.save("unet_model_clean.h5")

# -----------------------------------
# 6. CLI Entry Point
# -----------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nowcasting radar forecasting script")
    parser.add_argument("--data_path", type=str, default="radar_data.nc", help="Path to radar NetCDF file")
    parser.add_argument("--model", type=str, choices=["baseline", "unet"], default="baseline", help="Model type")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for DL model")
    args = parser.parse_args()
    main(args)