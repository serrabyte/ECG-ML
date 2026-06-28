# ECG-ML
# ECG Signal Classification Using DL Models
A Biomedical Engineering graduation project that leverages deep learning to automatically detect and classify cardiac arrhythmias from ECG signals.

## Overview

Cardiac arrhythmia is a potentially life-threatening condition characterized by irregular heart rhythms. Traditional diagnosis requires a specialist to manually analyze ECG recordings, which is time-consuming and resource-intensive.

This project automates arrhythmia classification using three deep learning architectures trained on the MIT-BIH Arrhythmia Database.


## Models Implemented

| Model | Train Accuracy | Train AUC | Validation Accuracy | Validation AUC |
|-------|---------------|-----------|---------------------|----------------|
| Dense Neural Network (DNN) | 97.8% | 99.5% | 96.9% | 98.8% |
| Convolutional Neural Network (CNN) | 96.8% | 99.1% | 96.1% | 98.6% |
| Long Short-Term Memory (LSTM) | 69.3% | 59.3% | 69.5% | 58.0% |




## Project Structure

```
├── data/                  # ECG data loaded via wfdb from MIT-BIH
├── models/
│   ├── dnn_model.py       # Dense Neural Network
│   ├── cnn_model.py       # Convolutional Neural Network
│   └── lstm_model.py      # Bidirectional LSTM
├── preprocessing/
│   ├── load_ecg.py        # Load records and annotations
│   └── preprocess.py      # Beat segmentation & labeling
├── evaluation/
│   └── metrics.py         # Accuracy, AUC, ROC curves, confusion matrices
├── notebooks/             # Google Colab notebooks
└── README.md
```



## Methodology

### 1. Data Acquisition
- Dataset: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) via PhysioNet
- 48 half-hour two-channel ambulatory ECG recordings from 47 patients (1975–1979)
- Accessed via Python's `wfdb` library

### 2. Preprocessing
- Heartbeats categorized as **normal (N)** or **abnormal**; non-beat symbols excluded
- Labels: `0` = normal, `1` = abnormal (binary classification)
- Each beat segmented with a fixed window of milliseconds before/after the beat

### 3. Feature Extraction
- Raw ECG signal segments used directly as input features
- No handcrafted feature engineering — the models learn representations automatically

### 4. Data Splitting
- **80% training** / **20% testing** split
- Validation set carved from the training set for hyperparameter tuning

### 5. Model Training

**DNN**
- 1 hidden layer, 32 neurons, ReLU activation
- Dropout (rate=0.25), Sigmoid output
- Optimizer: Adam | Loss: Binary Cross-Entropy | Epochs: 10 | Batch size: 32

**CNN**
- 1D Conv layer with 128 filters, kernel size 5, ReLU activation
- Dropout (rate=0.25), Flatten → Dense Sigmoid output
- Optimizer: Adam | Loss: Binary Cross-Entropy | Epochs: 2 | Batch size: 32

**LSTM**
- Bidirectional LSTM with 64 units
- Dropout (rate=0.25), Dense Sigmoid output
- Optimizer: Adam | Loss: Binary Cross-Entropy | Epochs: 1 | Batch size: 32
- Trained on a 10,000-sample subset due to computational constraints

### 6. Evaluation
Metrics: **Accuracy, Precision, Recall, Specificity, AUC, ROC Curve, Confusion Matrix**



##  Getting Started

### Prerequisites
```bash
pip install wfdb numpy pandas matplotlib scikit-learn tensorflow keras
```

### Run in Google Colab 
The project was developed and tested on [Google Colaboratory](https://colab.research.google.com/) for cloud-based processing of large ECG datasets.

### Local Setup
```bash
git clone https://github.com/your-username/arrhythmia-classification.git
cd arrhythmia-classification
pip install -r requirements.txt
```



## Results Summary

- **DNN** achieved the best overall performance with 97.8% training accuracy and 99.5% AUC
- **CNN** performed comparably and benefits from learning spatial patterns in ECG waveforms
- **LSTM** underperformed, likely due to training on a reduced subset and limited epochs
- ROC curves for all models are plotted for comparison on the validation set



##  Clinical Impact

| Area | Impact |
|------|--------|
| **Speed** | Near-instant diagnosis vs. hours of specialist review |
| **Accessibility** | Usable by a single trained nurse — no cardiologist needed at point of capture |
| **Cost** | Reduces patient consultation fees and repeat testing |
| **Safety** | Minimizes patient-specialist contact (critical during outbreaks like COVID-19) |
| **Environment** | Eliminates paper ECG printouts and reduces resource waste |



##  Limitations

- Trained solely on the MIT-BIH dataset — may not generalize to all real-world arrhythmia variants
- LSTM performance was limited by computational constraints (subset training, single epoch)
- Models are not interpretable enough yet for standalone clinical decision-making
- No real-time deployment pipeline implemented



##  Future Work

- Replace LSTM with a more suitable sequential model or extend training budget
- Explore ensemble methods combining DNN + CNN predictions
- Implement hyperparameter optimization (e.g., Optuna, Keras Tuner)
- Validate on larger, more diverse ECG databases
- Develop a real-time arrhythmia monitoring interface
- Address model interpretability (e.g., SHAP, Grad-CAM for time series)
- Ensure compliance with HIPAA, GDPR, and UAE Personal Data Protection Law



##  Dataset

**MIT-BIH Arrhythmia Database**
- Source: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- 48 recordings, 47 patients, sampled at 360 Hz
- Annotations include beat type labels used for supervised learning



##  Tech Stack

- **Language:** Python
- **Platform:** Google Colaboratory
- **Libraries:** TensorFlow / Keras, NumPy, Matplotlib, Scikit-learn, WFDB

