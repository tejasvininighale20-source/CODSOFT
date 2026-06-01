# 🎬 Movie Rating Prediction with Python

## CODSOFT Data Science Internship – Task 2

### 📌 Project Overview

This project focuses on predicting IMDb movie ratings using Machine Learning techniques. The dataset contains information about Indian movies, including their release year, duration, genres, directors, actors, and voting statistics.

The project follows a complete Data Science workflow, including data cleaning, feature engineering, exploratory data analysis (EDA), model training, model evaluation, cross-validation, and feature importance analysis.

The final model uses **XGBoost Regressor**, which achieved the best performance among the tested algorithms.

---

## 📂 Dataset

**Dataset:** IMDb Movies India Dataset

### Features Available

* Name
* Year
* Duration
* Genre
* Rating (Target Variable)
* Votes
* Director
* Actor 1
* Actor 2
* Actor 3

---

## 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Pickle

---

## 🔄 Project Workflow

### 1. Data Preprocessing

* Removed records with missing ratings
* Extracted numeric values from Year and Duration
* Converted Votes into numerical format
* Handled missing values using median imputation

### 2. Feature Engineering

Created additional features to improve model performance:

* Log-transformed Votes (LogVotes)
* Movie Decade
* Genre Count
* Primary Genre
* Director Frequency Encoding
* Actor Frequency Encoding
* Cast Popularity Score
* Genre-Based Features

### 3. Exploratory Data Analysis (EDA)

Generated visualizations to understand the dataset:

* Rating Distribution
* Votes vs Rating Relationship
* Top Directors Analysis
* Genre-wise Average Ratings
* Correlation Heatmap
* Rating Trends Across Years
* Most Voted Movies
* Most Active Actors

### 4. Machine Learning Models

The following regression models were trained and compared:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

### 5. Model Evaluation

Performance was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### 6. Model Saving

The best-performing model was saved using Pickle for future predictions.

---

## 📊 Results

After comparing multiple models using Cross-Validation, **XGBoost Regressor** delivered the best overall performance and was selected as the final model.

Evaluation Metrics:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## ⭐ Key Features

* Comprehensive Data Cleaning
* Advanced Feature Engineering
* Exploratory Data Analysis (EDA)
* Cross-Validation for Model Comparison
* Multiple Regression Models
* XGBoost-Based Prediction System
* Feature Importance Visualization
* Model Serialization

---

## 👩‍💻 Author

**Tejasvini Nighale**

CODSOFT Data Science Internship

