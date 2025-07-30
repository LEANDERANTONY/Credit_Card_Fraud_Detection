# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions in a highly imbalanced dataset.

## 🔍 Problem Statement
Develop a machine learning model that accurately classifies fraud cases from legitimate transactions using anonymized data.

## 📁 Data
Dataset used: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
Contains 284,807 transactions with 492 fraud cases (~0.17%)


## ⚙️ Approach
- Preprocessing: Dropped 'Time', scaled 'Amount'
- Handled class imbalance using SMOTE and ADASYN
- Models: Logistic Regression, Random Forest, XGBoost
- Cross-validation with stratified folds
- ROC-AUC used as primary metric

## 🏆 Results
- Best ROC-AUC: ~0.977 (XGBoost + threshold tuning)
- Balanced precision/recall with custom threshold

## 🧰 Tech Stack
`Python`, `Pandas`, `Scikit-learn`, `XGBoost`, `Imbalanced-learn`, `Matplotlib`, `Seaborn`
