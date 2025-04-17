# FidelFolio_DL
# Financial Time Series Forecasting with Deep Learning

## Overview

This repository contains Python scripts for forecasting financial metrics for multiple companies over time (panel data). It implements several deep learning approaches, including Long Short-Term Memory (LSTM) networks (in different configurations), Multi-Layer Perceptrons (MLP), and Encoder-Decoder LSTMs, to predict multiple target variables. The code employs an expanding window cross-validation methodology suitable for time series data.

## Key Features

*   **Panel Data Handling:** Processes data containing multiple companies across multiple years.
*   **Multi-Target Prediction:** Designed to predict several target variables (e.g., `Target 1`, `Target 2`, `Target 3`).
*   **Multiple Model Architectures:**
    *   **Base LSTM:** A standard stacked LSTM model.
    *   **Improved LSTM:** Incorporates feature engineering (YoY differences) and PCA before feeding data to a stacked LSTM.
    *   **MLP:** Uses a Multi-Layer Perceptron, treating the flattened historical sequence as input features.
    *   **Encoder-Decoder LSTM:** Uses an LSTM encoder to create a context vector from the historical sequence, which is then used by a dense decoder part to make predictions.
*   **Expanding Window Cross-Validation:** Simulates a realistic forecasting scenario where the model is retrained periodically with accumulating historical data.
*   **Per-Target Model Training:** Trains a separate, specialized model for each target variable within each cross-validation fold.
*   **Data Preprocessing Pipeline:**
    *   Column Name Cleaning
    *   Automatic Numeric Conversion
    *   Feature Engineering (Year-over-Year Differences)
    *   KNN Imputation for Missing Values
    *   Robust Scaling for Features
    *   Outlier Capping (optional, applied in some versions)
    *   Principal Component Analysis (PCA) for dimensionality reduction (optional, applied in some versions)
    *   Standard Scaling for Target Variables
    *   Company ID Label Encoding
*   **Sequence Preparation:** Transforms time series data into padded sequences suitable for RNN input.
*   **Model Training:** Uses TensorFlow/Keras with callbacks for Early Stopping and Learning Rate Reduction.
*   **Evaluation:** Calculates and reports Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on the original scale for each target variable, aggregated across all test folds.
*   **Configuration:** Key hyperparameters and settings are defined at the beginning of the script for easy modification.

## Requirements

*   Python 3.8+
*   pandas
*   numpy
*   scikit-learn
*   tensorflow (>= 2.x)
*   matplotlib (optional, for visualizations)
*   seaborn (optional, for visualizations)

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
