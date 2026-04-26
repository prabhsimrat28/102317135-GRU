# ⚡ GRU-Based Time Series Forecasting — Electric Production

## 📌 Project Overview

This project implements a **GRU (Gated Recurrent Unit)** neural network using PyTorch to forecast electric production from time series data.

### Features:
- Time series preprocessing
- Sliding window dataset creation
- GRU-based model training
- Model evaluation using MSE
- Ablation study on window sizes
- Visualization of results

---

## 🧠 Model Architecture

- Model: GRU (Gated Recurrent Unit)
- Framework: PyTorch

### Why GRU?
- Handles sequential data effectively
- Reduces vanishing gradient problem
- Faster than LSTM due to fewer parameters

---

## ⚙️ Hyperparameters

- Window Size: Input sequence length
- Horizon: Number of future steps to predict
- Hidden Size: GRU hidden layer size
- Learning Rate: Optimizer step size
- Epochs: Number of training iterations

---

## 📂 Dataset

- File: `Electric_Production.csv`

### Columns:
- `date`: Timestamp
- `value`: Electric production values

---

## 🔄 Workflow

### 1. Data Preprocessing
- Convert date column to datetime
- Normalize values
- Prepare sequences

### 2. Sliding Window Creation
Transforms time series into:
- Input: past `window_size` values
- Output: next `horizon` values

### 3. Model Training
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam

### 4. Evaluation
- Predict on test data
- Calculate MSE

---

## 🧪 Ablation Study

Tested configurations:
- Smaller window size
- Original window size
- Larger window size

### Objective:
Evaluate impact of input sequence length on performance

---

## 📊 Results

- Training loss plotted over epochs
- Predictions vs actual values visualization
- Comparison table of MSE scores

---

## ⚠️ Failure Analysis

- Identifies high-error predictions
- Highlights model limitations

---

## 🛠️ Installation

```bash
pip install numpy pandas matplotlib torch scikit-learn
