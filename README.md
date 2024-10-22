# Typhoon Path Prediction

> A Typhoon Path Prediction Model Implemented Using LSTM

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Feature Selection](#feature-selection)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [License](#license)

---

## Overview

This project provides a complete workflow for predicting typhoon paths using an LSTM-based model. The model predicts the typhoon's path by using data from the previous 4 time points to forecast the next time point. The workflow includes data preparation, feature extraction, model training, prediction, and visualization of results.

---

## Installation

### Prerequisites:
- Python 3.x
- `pip` package manager

### Steps:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/veraleiwengian/typhoon-path-prediction.git
   ```

2. **Navigate into the repository:**
   ```bash
   cd typhoon-path-prediction
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Data preparation

We use the [CMA Tropical Cyclone Best Track Dataset](https://tcdata.typhoon.org.cn/en/zjljsjj.html).

To clean and prepare the dataset, run:
   ```bash
   python3 data_clean.py
   ```

**Modifications:**

- `END` column: Records the status of the tropical cyclone at the current time point:
    - 0: Not ended
    - 1: Dissipated
    - 2: Moved out of the Western Pacific Typhoon Committee’s responsibility area
    - 3: Merged
    - 4: Quasi-stationary
- `distance_km` and `bearing` columns: These represent the Haversine distance and bearing angle from the previous time point to the current time point, providing more spatial information about the cyclone’s movement.

---

## Feature selection

We enhance the model’s feature representation with the following techniques:

- **Temporal Features**: Convert the Time column into cyclical representations (hour, day, month, year) using sine and cosine transformations to capture time-based cyclical patterns.

- **Bearing**: The bearing angle is also transformed using sine and cosine to better capture angular relationships.

- **Categorical Features**:
    - I (Intensity) and END (Status) are one-hot encoded to allow the model to differentiate between different categories.

- **Normalization**:
    - Remaining numerical columns are normalized using min-max scaling to standardize the data and improve model training efficiency.

---

## Model training

To train the model:
   ```bash
   python3 train.py
   ```


**Training Details**:

- **Loss function**: SmoothL1Loss (Huber Loss), chosen for its robustness against outliers, balancing between L1 and L2 loss.

- **Optimizer**: AdamW, which combines the benefits of Adam (adaptive learning rates) with weight decay to prevent overfitting and improve generalization.

- **Data split**:
    - 90% of the dataset is used for training.
    - 10% is reserved for validation.

- The best model is automatically saved based on the validation loss.

---

## Prediction

To make predictions:
   ```bash
   python3 predict.py
   ```

- You can specify a typhoonID to run predictions.
- The model uses the previous 4 time points of data to predict the next time point’s typhoon data.
- After prediction, a plot comparing the actual vs. predicted data is generated.

Example plot:
![comparison plot](./result/plot_20230019_11.jpg)

---

## License

This project is licensed under the **MIT License**.