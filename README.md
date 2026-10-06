# Credit-Card 
# 💳 Credit Card Default Prediction

## 📌 Project Overview

This project uses **Machine Learning classification techniques** to predict whether a credit card customer will **default on their next month's payment**.

The project includes data preprocessing, exploratory data analysis, categorical encoding, outlier handling, SMOTE-based class balancing, feature selection, feature scaling, and multiple machine learning classification algorithms.

---

## 🎯 Objectives

- To analyze credit card customer payment and demographic information.
- To predict whether a customer will default on their next month's payment.
- To compare different machine learning classification algorithms based on their performance.

---

## 📊 Dataset

The project uses the **Credit Card Default** dataset stored as:

`creditcard.csv`

The dataset contains information related to customers' demographics, credit limits, previous payment status, bill amounts, and payment amounts.

### Target Variable

**`target`**

The original column:

`default payment next month`

was renamed to `target`.

The target values were converted into:

- `YES` → Customer defaults
- `NO` → Customer does not default

---

## 🔍 Exploratory Data Analysis

The project performs several initial data analysis steps:

- Displaying the first and last records
- Checking dataset information
- Checking dataset shape
- Examining column names
- Generating descriptive statistics
- Checking missing values
- Checking duplicate records
- Creating a correlation matrix
- Visualizing correlations using heatmaps
- Detecting and visualizing outliers using boxplots

---

## 🧹 Data Preprocessing

Several preprocessing steps were performed:

### Column Renaming

Payment, bill, and payment amount columns were renamed to make them easier to understand.

Examples:

- `PAY_0` → `SEP_PAY`
- `PAY_2` → `AUG_PAY`
- `BILL_AMT1` → `SEP_BILL`
- `PAY_AMT1` → `SEP_PAYMENT`

The target column was renamed:

`default payment next month` → `target`

### Categorical Data Conversion

Categorical values were converted into meaningful labels.

For example:

**SEX**
- `1` → `M`
- `2` → `F`

**MARRIAGE**
- `1` → `MARRIED`
- `2` → `SINGLE`
- `3` / `0` → `OTHERS`

**EDUCATION**
- `1` → `UG`
- `2` → `PG`
- `3` → `HIGH_SCHOOL`
- Other values → `OTHERS`

---

## 📦 Outlier Handling

The project uses the **IQR (Interquartile Range)** method to detect and handle numerical outliers.

A custom function named:

`fix_outliers_iqr()`

was created to replace values outside the calculated lower and upper bounds.

---

## ⚖️ Handling Class Imbalance

The project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the target classes.

```python
from imblearn.over_sampling import SMOTE
```

SMOTE is applied before model training so that the classes are more balanced.

---

## 🔤 Encoding

The project uses:

### Label Encoding

`LabelEncoder` is used for:

- `target`
- `SEX`

### One-Hot Encoding

`OneHotEncoder` is used for:

- `EDUCATION`
- `MARRIAGE`

This converts categorical variables into numerical features suitable for machine learning models.

---

## 🎯 Feature Selection

The project uses:

**SelectKBest with ANOVA F-test**

to select the **25 best features**.

```python
SelectKBest(score_func=f_classif, k=25)
```

This helps reduce the number of features used for model training.

---

## 📏 Feature Scaling

The selected features are standardized using:

**StandardScaler**

```python
StandardScaler()
```

The data is then divided into training and testing sets using:

- **80% Training Data**
- **20% Testing Data**

with:

```python
random_state=40
```

---

## 🤖 Machine Learning Models

The project compares the following classification algorithms:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **AdaBoost Classifier**
5. **Gradient Boosting Classifier**

---

## 📈 Evaluation Metrics

The models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Classification Report**

These metrics are used to compare the performance of the different classification models.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook

---

## 📂 Project Structure

```text
Credit-Card-Default-Prediction/
│
├── credit card project.ipynb
├── creditcard.csv
└── README.md
```

---

## 🚀 Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Categorical Encoding
   ↓
Outlier Handling
   ↓
SMOTE
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Model Comparison
```

---

## 💡 Conclusion

This project demonstrates an end-to-end **machine learning classification workflow** for predicting credit card payment default.

Multiple classification algorithms are trained and evaluated using accuracy, precision, recall, and F1 score to identify their performance on the prediction task.
