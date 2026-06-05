# 🌸 Iris Flower Classification

## 📌 Project Overview

The Iris Flower Classification project is a Machine Learning classification task that predicts the species of an Iris flower using sepal and petal measurements.

The model classifies flowers into one of the following categories:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

This project was developed as part of the CodSoft Data Science Internship and demonstrates a complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, hyperparameter tuning, evaluation, and prediction.

---

## 🎯 Project Objective

The objective of this project is to build an accurate machine learning model capable of identifying Iris flower species based on their morphological features.

---

## 📂 Dataset Information

The dataset contains flower measurements with the following features:

### Input Features

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Target Variable

* Species

### Dataset Source

https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

* Data Loading
* Duplicate Record Detection and Removal
* Missing Value Analysis
* Statistical Summary Generation
* Label Encoding
* Feature and Target Separation
* Train-Test Split

---

## 📊 Exploratory Data Analysis (EDA)

Several visualizations were created to understand the dataset and feature relationships:

* Species Distribution Plot
* Pair Plot Analysis
* Correlation Heatmap
* Feature Distribution Boxplots
* Feature Importance Visualization

These visualizations helped identify class distributions, feature correlations, and important predictors.

---

## 🤖 Machine Learning Models

The following classification models were trained and compared:

### Logistic Regression

A simple and efficient linear classification algorithm.

### Support Vector Machine (SVM)

A powerful classification algorithm that identifies optimal decision boundaries between classes.

### Random Forest Classifier

An ensemble learning technique that combines multiple decision trees to improve prediction performance and robustness.

---

## 📈 Model Evaluation

Model performance was evaluated using:

* Accuracy Score
* Cross Validation Accuracy
* Classification Report
* Confusion Matrix
* Cohen's Kappa Score

### Validation Technique

* Stratified K-Fold Cross Validation (5-Fold)

This validation strategy ensures reliable and unbiased model performance assessment.

---

## ⚙️ Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest model by testing multiple hyperparameter combinations, including:

* Number of Estimators
* Maximum Tree Depth
* Minimum Samples Split

The best-performing parameter combination was selected automatically based on cross-validation accuracy.

---

## 🏆 Model Comparison and Selection

All models were evaluated using cross-validation scores and ranked according to their performance.

A comparison table and visualization were generated to identify the best-performing classifier.

---

## 🌟 Feature Importance Analysis

Feature importance scores were extracted from the optimized Random Forest model to determine which flower measurements contributed most to classification.

Both importance values and percentage contributions were analyzed and visualized.

---

## 💾 Model Saving

The final trained model and label encoder were saved using Joblib for future predictions.

### Generated Files

* iris_classifier.pkl
* label_encoder.pkl

---

## 🔮 Sample Prediction

The trained model can predict the species of a new Iris flower based on user-provided measurements.

In addition to the predicted species, the model also provides probability scores for each class, helping interpret prediction confidence.

---

## 🚀 Key Learning Outcomes

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Data Visualization
* Classification Algorithms
* Stratified K-Fold Cross Validation
* Hyperparameter Tuning with GridSearchCV
* Model Comparison and Selection
* Classification Metrics Evaluation
* Cohen's Kappa Score
* Feature Importance Analysis
* Probability-Based Predictions
* Model Persistence using Joblib

---

## 👩‍💻 Author

**Tejasvini Nighale**

Data Science Intern @ CodSoft

