# Vaccine Cold-Chain Risk Prediction Using Federated Learning

### A Comparative Study of Centralized Learning, FedAvg and FedProx

## 1. Project Overview

Maintaining the appropriate temperature conditions during vaccine storage and transportation is essential for preserving vaccine quality and ensuring effective immunization.

This project explores the application of Federated Learning (FL) for predicting vaccine cold-chain risks across multiple Primary Health Centres (PHCs).

The system simulates five independent PHC clients, each maintaining its own local dataset. Instead of combining all local data for centralized training, federated learning allows the clients to collaboratively train a shared global model through model parameter aggregation.

The project implements and compares three approaches:
- Centralized Machine Learning
- Federated Averaging (FedAvg)
- Federated Proximal (FedProx)

The objective is to classify vaccine cold-chain conditions into three categories: Low, Moderate, and High risk.

## 2. Project Objectives

- Develop a machine learning model for vaccine cold-chain risk classification.
- Simulate distributed healthcare environments using five PHC clients.
- Implement Federated Averaging (FedAvg) and Federated Proximal (FedProx).
- Investigate the impact of Non-IID data distribution on model performance.
- Compare centralized and federated learning approaches.
- Evaluate model performance using accuracy, precision, recall and F1-score.
- Conduct feature ablation experiments to study the contribution of additional cold-chain parameters.

## 3. Dataset Description

The project uses synthetically generated vaccine cold-chain data representing five Primary Health Centres.

Each PHC acts as an independent federated learning client with its own local dataset.

### Input Features

| Feature | Description |
|---|---|
| Temperature_C | Vaccine storage temperature |
| Humidity_percent | Storage humidity |
| Storage_Duration_hours | Duration of vaccine storage |
| Transport_Duration_hours | Transportation duration |
| Num_Excursions | Number of temperature excursions |
| Excursion_Duration_hours | Duration of excursions |
| Temperature_Deviation_C | Deviation from the acceptable temperature range |
| Cumulative_Exposure | Cumulative temperature exposure |

**Target Variable:** Risk_Level

The target consists of three categories:
- Low
- Moderate
- High

The synthetic dataset considers the recommended vaccine storage temperature range of 2°C–8°C and simulates different operating conditions across PHCs to create heterogeneous data distributions.

## 4. Methodology

### 4.1 Data Generation and Preprocessing

Synthetic cold-chain datasets are generated for five PHC clients. Data preprocessing includes feature selection, train-test splitting, and feature standardization using StandardScaler.

### 4.2 Centralized Learning

A centralized Multi-Layer Perceptron (MLP) model is trained using the combined dataset. This serves as the baseline for comparison with federated learning approaches.

### 4.3 Federated Averaging (FedAvg)

FedAvg enables five PHC clients to train the same global model collaboratively. Each client performs local training, and its model weights are sent for aggregation. The server combines the weights using a sample-size-weighted averaging approach to update the global model.

### 4.4 Federated Proximal (FedProx)

FedProx extends the FedAvg approach by introducing a proximal regularization term to reduce the impact of differences between local client data distributions.

The implementation uses a proximal parameter of μ = 0.01.

## 5. Model Architecture

The project uses a Multi-Layer Perceptron (MLP) neural network.

| Layer | Configuration |
|---|---|
| Input Layer | 8 numerical features |
| Hidden Layer 1 | 32 neurons, ReLU |
| Dropout | 0.2 |
| Hidden Layer 2 | 16 neurons, ReLU |
| Dropout | 0.2 |
| Output Layer | 3 neurons, Softmax |

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Cross-Entropy
- Batch Size: 32
- Federated Communication Rounds: 20
- Local Training Epochs: 3
- Number of Clients: 5
- FedProx μ: 0.01

## 6. Additional Experiments

The project also includes:

- Random Forest classification.
- XGBoost classification.
- Feature ablation experiments.
- Comparison of model performance with different feature combinations.
- Training convergence analysis.
- PHC-wise data distribution visualizations.
- Confusion matrix and performance evaluation.

The ablation experiments investigate how additional parameters, such as power outage duration and door-opening frequency, affect prediction performance.

## 7. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Convergence

Experimental results and training histories are saved as CSV files, while visualizations are stored in the figures directories.

Refer to the results and ablation folders for the recorded experimental outputs.

## 8. Project Structure

```text
Vaccine-Cold-Chain-Federated-Learning/
│
├── data/
│   ├── PHC1.csv
│   ├── PHC2.csv
│   ├── PHC3.csv
│   ├── PHC4.csv
│   ├── PHC5.csv
│   └── combined_dataset.csv
│
├── data_v2/
│
├── data_ablation/
│
├── figures/
│
├── figures_v2/
│
├── ablation/
│   ├── centralized_ablation.py
│   ├── fedavg_ablation.py
│   ├── plot_centralized_ablation.py
│   └── ablation performance results
│
├── centralized_mlp.py
├── centralized_mlp_v2.py
├── fedavg.py
├── fedavg_v2.py
├── fedprox.py
├── random_forest.py
├── xgboost_model.py
├── generate_dataset.py
├── generate_dataset2.py
├── add_ablation_features.py
└── check_datasets.py
```

The repository also contains saved model files, training histories, evaluation outputs and visualization results.

## 9. Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Federated Learning
- Neural Networks
- Random Forest
- XGBoost

## 10. Research Contribution

This project investigates the use of Federated Learning for collaborative vaccine cold-chain risk prediction across distributed healthcare facilities.

By simulating independent PHC clients, the study explores how decentralized model training can support healthcare risk classification without requiring raw local datasets to be combined into a single centralized training dataset.

The comparison of FedAvg and FedProx also provides insights into the challenges of heterogeneous data distributions in federated healthcare applications.

## 11. Disclaimer

This is an academic research project based on synthetically generated data. It is intended for experimental and educational purposes and is not a validated clinical decision-support system or a substitute for actual vaccine storage monitoring.

## 12. Author

Developed as an academic research project exploring the application of Federated Learning and Machine Learning in healthcare cold-chain risk prediction.
