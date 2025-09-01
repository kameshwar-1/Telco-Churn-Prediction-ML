# Telco Customer Churn Prediction:-

Predict whether a telecom customer will "churn" next month using classical ML.  
This project uses a clean "Logistic Regression pipeline" with one-hot encoding, scaling, and regularization tuned by cross-validation.


# Dataset:
Use the Telco Churn CSV  
url="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Method:
1. Cleaning:
   - Normalize column names, trim spaces, lowercase strings
   - Convert 'totalcharges' - numeric
   - Encode target 'churn: yes-1, no-0'
2. Split (stratified): Train 60% / Val 20% / Test 20%
3. Pipeline:
   - 'OneHotEncoder' for categoricals
   - 'StandardScaler' for numerics
   - 'LogisticRegression(class_weight="balanced")'
4. "Tune" 'C' via 'GridSearchCV' on "ROC-AUC"
5. "Refit" on Train+Val; "final eval"on Test



# Final Test Results 
Refitting final model with C = 10

Accuracy : 0.739
Precision: 0.506
Recall  : 0.829
F1 : 0.628
ROC-AUC : 0.847
Confusion Matrix:[[730 303][ 64 310]]


# How to Run:


# Clone repository
git clone https://github.com/kameshwar-1/Telco-Churn-Prediction-ML.git
cd Telco-Churn-Prediction-ML


# Run script

Telco-Churn-Prediction-ML.ipynb
