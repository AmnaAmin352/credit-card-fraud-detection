# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions in a highly imbalanced dataset, where fraud makes up only 0.167% of all transactions.

## 📊 Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions, only 473 (0.167%) labeled as fraud
- Features are anonymized (PCA-transformed) except `Time` and `Amount`

## 🛠️ Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## 🔍 Project Workflow
1. Exploratory Data Analysis, including class imbalance visualization
2. Feature scaling
3. Training and comparing 6 classification models
4. Evaluating with Precision, Recall, F1-score, ROC-AUC, and **PR-AUC** (chosen over raw accuracy due to extreme imbalance)
5. Hyperparameter tuning with GridSearchCV
6. Cross-validation (Stratified K-Fold)
7. Saving the best model with `joblib`

## 📈 Model Comparison (Fraud Class Metrics)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| **Random Forest (Tuned)** | **95.8%** | 72.6% | **0.826** | **0.815** |
| Random Forest | 97.1% | 70.5% | 0.817 | 0.796 |
| KNN | 97.1% | 70.5% | 0.817 | 0.777 |
| Decision Tree | 73.5% | 64.2% | 0.685 | 0.473 |
| Linear SVM | 7.1% | 87.4% | 0.131 | 0.648 |
| Logistic Regression | 5.6% | 87.4% | 0.106 | 0.672 |

**Key insight:** Since accuracy is misleading on imbalanced data (a model predicting "no fraud" every time is still ~99.8% accurate), **PR-AUC** was used as the primary metric. The tuned Random Forest achieved the best balance of precision and recall.

## 📁 Files
- `credit_card_fraud_detection.ipynb` — main notebook with EDA, model training, tuning, and evaluation

## 🚀 How to Run
1. Clone this repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project folder
3. Install requirements: `pip install pandas numpy scikit-learn matplotlib seaborn joblib`
4. Open the notebook in Jupyter Notebook and run all cells
