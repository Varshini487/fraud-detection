# 🔒 Fraud Detection System

A **real-time Fraud Detection system** using ML and anomaly detection to flag suspicious transactions.

## 🛡️ Detection Methods
| Method | Type | Best For |
|--------|------|---------|
| Isolation Forest | Unsupervised | Unknown fraud patterns |
| Autoencoder | Deep Learning | Reconstruction anomalies |
| XGBoost | Supervised | Labeled fraud data |
| SMOTE | Preprocessing | Class imbalance |

## ⚡ Performance
- Response time: <100ms per transaction
- False positive rate: <2%
- Fraud recall: >95%

## 🛠️ Tech Stack
- **Python, Scikit-learn, XGBoost**
- **TensorFlow** – Autoencoder
- **FastAPI** – Real-time API
- **Streamlit** – Dashboard

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/fraud-detection
cd fraud-detection
pip install -r requirements.txt
streamlit run app.py
```
