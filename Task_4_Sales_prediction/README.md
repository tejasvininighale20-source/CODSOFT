# Sales Prediction Using Python

## Overview

This project focuses on predicting product sales based on advertising investments across multiple marketing channels, including TV, Radio, and Newspaper. Using machine learning techniques, the project analyzes historical advertising data to identify patterns and estimate future sales performance.

The solution demonstrates an end-to-end data science workflow, covering data preprocessing, exploratory data analysis, feature engineering, model development, evaluation, and prediction.

## Business Problem

Organizations invest significant resources in advertising campaigns across different media platforms. Understanding how advertising expenditure influences sales is essential for optimizing marketing budgets and maximizing return on investment (ROI).

This project aims to answer the following questions:

* Which advertising channel has the greatest impact on sales?
* Can future sales be predicted based on advertising spending?
* Which machine learning model provides the most accurate predictions?

## Dataset

The dataset contains advertising expenditure information across three marketing channels:

* TV Advertising Budget
* Radio Advertising Budget
* Newspaper Advertising Budget

The target variable is:

* Sales

## Methodology

### 1. Data Preparation

* Loaded and inspected the dataset.
* Removed unnecessary columns.
* Verified data quality by checking for missing values and duplicate records.

### 2. Exploratory Data Analysis

Comprehensive analysis was performed to understand the dataset and identify relationships between variables:

* Distribution Analysis
* Outlier Detection
* Correlation Analysis
* Pairwise Feature Relationships
* Sales Trend Visualization

### 3. Feature Engineering

A new feature, **Total Advertising**, was created by combining advertising expenditures across all channels to capture overall marketing investment.

### 4. Model Development

Two regression models were implemented and evaluated:

* Linear Regression
* Random Forest Regressor

### 5. Model Evaluation

Model performance was assessed using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* 5-Fold Cross Validation

### 6. Performance Analysis

Additional evaluation techniques included:

* Actual vs Predicted Sales Visualization
* Residual Error Analysis
* Feature Importance Analysis
* Comparative Model Performance Assessment

## Results

The developed models successfully learned the relationship between advertising expenditure and sales performance. Comparative analysis demonstrated the predictive capabilities of both Linear Regression and Random Forest models, while feature importance analysis provided insights into the advertising channels that contribute most significantly to sales growth.

## Key Insights

* Advertising expenditure exhibits a strong relationship with sales performance.
* Certain advertising channels contribute more significantly to sales than others.
* Machine learning models can effectively support marketing budget allocation and sales forecasting.

## Conclusion

This project demonstrates the practical application of machine learning in sales forecasting and marketing analytics. By leveraging historical advertising data, businesses can make more informed decisions regarding advertising strategies, budget optimization, and revenue forecasting.

---

**CodSoft Data Science Internship**

**Task 4 – Sales Prediction Using Python**

