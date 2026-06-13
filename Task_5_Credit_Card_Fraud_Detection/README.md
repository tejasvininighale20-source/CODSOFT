💳 Credit Card Fraud Detection Pipeline

📌 Project Overview

The Credit Card Fraud Detection project is an advanced Machine Learning classification task designed to identify anomalous and fraudulent credit card transactions in real time. 
The model classifies transactions into one of the following categories:
* Genuine Transaction (Class 0)
* Fraudulent Transaction (Class 1)

This project was developed as part of the CodSoft Data Science Internship and demonstrates a highly structured, error-free machine learning workflow optimized for execution speed, extreme class imbalance resolution, and automated business intelligence reporting.

🎯 Project Objective

The objective of this project is to build an exceptionally robust predictive model capable of maximizing fraud capture rates (Recall) and minimizing financial loss while mitigating false positives for genuine banking customers.

📂 Dataset Information

The dataset contains real-world credit card transactions containing anonymized features due to privacy requirements, resulting in principal components (V1-V28) obtained via PCA.
Input Features:
* Time (Seconds elapsed between each transaction and the first transaction)
* V1, V2, ... V28 (Anonymized features obtained via PCA)
* Amount (Transaction monetary value)

Target Variable:
* Class (0 for Genuine, 1 for Fraud)

Dataset Source:
* Managed and securely fetched via `kagglehub` from the `mlg-ulb/creditcardfraud` repository.

🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (`imblearn`)

🔍 Data Preprocessing & Feature Engineering

The following preprocessing, data cleaning, and custom engineering steps were performed:
* Automated Data Ingestion using headless fetching tools.
* Duplicate Record Detection and Removal to prevent data leakage.
* Missing Value Analysis to guarantee zero empty entries.
* Logarithmic Scaling (`np.log1p`) applied to transaction amounts to minimize severe data skewness.
* Temporal Feature Extraction (`Hour`) to track cyclical, time-based anomaly windows.
* Domain-Specific Threshold Flags to dynamically isolate `Micro_Transactions` and `High_Value_Transactions`.
* Feature Scaling using `StandardScaler` for standardized model inputs.

⚖️ Handling Class Imbalance

Real-world fraud datasets present an extreme data skewness challenge where fraud instances account for a mere **0.17%** of total data. To prevent algorithmic bias toward the majority class:
* **SMOTE (Synthetic Minority Over-sampling Technique)** was implemented strictly on the training partition to mathematically balance classes before modeling.

📊 Exploratory Data Analysis (EDA)

Several statistical visualizations were created to unpack hidden transactional signatures:
* Transaction Distribution Bar Plots (Mapping exact counts)
* Comparative Pie Charts (Visualizing class disparity percentages)
* Amount Distribution Boxplots (Highlighting extreme outliers between classes)

🤖 Machine Learning Models

The pipeline benches a linear algorithm against a high-capacity ensemble technique:
* **Logistic Regression:** A fast-converging, regularized linear classification algorithm serving as an efficient baseline.
* **Random Forest Classifier:** A parallelized ensemble learning technique structured with depth-limiting parameters to map complex, non-linear fraud patterns smoothly without overfitting.

📈 Model Evaluation

Because traditional accuracy is misleading in highly imbalanced datasets, model performance was evaluated using critical classification matrices:
* Accuracy Score
* Precision Score
* Recall (Sensitivity) — Optimized to ensure maximum true fraud capture.
* F1-Score — Balancing the trade-off between precision and recall.
* Confusion Matrix Analysis
* Classification Report Profiles

⚙️ Hyperparameter Tuning & Architecture Optimization

The Random Forest model was structured with specific performance constraints (`n_estimators=100`, `max_depth=8`) and executed with multi-core parallelization (`n_jobs=-1`) to maintain enterprise-grade execution speeds while processing hundreds of thousands of transactions.

📊 Model Comparison and Selection

A side-by-side comparative validation table was generated to systematically analyze the metrics of both models. While Logistic Regression provided high baseline sensitivity, the Random Forest model achieved superior metric stability across Precision and F1-Scores.

🌟 Feature Importance & Business Insights

* **Feature Importance Hierarchy:** Extracted and plotted the top 15 latent variable vectors driving anomalous activities to provide instant structural visibility for risk analysts.
* **Risk Capital Quantification:** Implemented business logic to automatically calculate and report the total **Potential Fraud Amount at Risk** across the dataset.

🔮 Sample Prediction

The pipeline concludes with a modular system interface capable of extracting an isolated transaction sample, applying saved scaling transformations, and outputting an instantaneous classification response (`Genuine` or `Fraud`).

🚀 Key Learning Outcomes

* Designing production-grade, error-free machine learning pipelines.
* Mitigating severe class imbalances using synthetic resampling wrappers (`SMOTE`).
* Formulating feature engineering architectures for financial transaction streams.
* Evaluating high-stakes fraud systems using Recall, Precision, and Confusion Matrices.
* Extracting structural feature importances to drive corporate strategy.

👩‍💻 Author
Tejasvini Nighale

Data Science Intern @ CodSoft
