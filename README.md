# Customer Churn Prediction Project

This project is an end-to-end machine learning solution designed to predict customer churn for a telecommunications company. The aim is to identify customers who are likely to cancel their subscription, enabling the business to implement retention strategies and reduce churn rates.

## Table of Contents
- [Introduction](#introduction)
- [Objective](#objective)
- [Dataset](#dataset)
- [Data Understanding & Preprocessing](#data-understanding--preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Deployment](#model-deployment)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results and Insights](#results-and-insights)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Customer churn refers to the rate at which customers leave a product or service. Predicting churn is crucial for subscription-based businesses, as retaining existing customers is often more cost-effective than acquiring new ones.

## Objective

The objective of this project is to build a machine learning model that can accurately predict whether a customer will churn, based on various customer attributes. This model can help the business in identifying high-risk customers and taking targeted actions to improve retention.

## Dataset

The project uses the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn), which includes:
- **Rows**: 7,043 customer records
- **Columns**: 21 features including customer demographics, account information, services used, and whether they churned

## Data Understanding & Preprocessing

### Key Steps:
- Identified and handled missing values, specifically in the `TotalCharges` column.
- Converted categorical variables into numerical representations using one-hot encoding and label encoding.
- Standardized continuous variables such as `tenure`, `MonthlyCharges`, and `TotalCharges` to ensure better model performance.

## Exploratory Data Analysis

Performed EDA to uncover key insights:
- **Churn Rate**: Customers with month-to-month contracts are more likely to churn.
- **Service Usage**: Customers with fiber optic internet tend to have a higher churn rate compared to DSL users.
- **Tenure**: Longer-tenure customers are less likely to churn.

Several visualizations were created using `matplotlib` and `seaborn` to identify patterns, correlations, and trends in the data.

## Feature Engineering

- **Feature Selection**: Identified important features such as `Contract`, `MonthlyCharges`, and `tenure` that contribute significantly to predicting churn.
- **Feature Encoding**: Applied one-hot encoding for categorical variables and scaled numerical features using `StandardScaler`.

## Model Building and Evaluation

Built and evaluated several machine learning models:
- **Logistic Regression**: Baseline model with 80.5% accuracy.
- **Random Forest**: Improved performance after tuning, with a 78.6% accuracy.
- **XGBoost**: Achieved the highest accuracy of 78.2% with an ROC-AUC score of 81.7%.

Metrics used for evaluation:
- Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

## Hyperparameter Tuning

Employed `GridSearchCV` to fine-tune hyperparameters for the Random Forest and XGBoost models. This step significantly improved the models' recall and F1 score, demonstrating the importance of hyperparameter optimization in model performance.

### Example Tuning Parameters:
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`
- **XGBoost**: `learning_rate`, `max_depth`, `n_estimators`, `subsample`

## Model Deployment

The final XGBoost model was deployed using Flask, providing a REST API endpoint that allows users to make real-time predictions.

### Deployment Steps:
1. **API Development**: Created a Flask app (`app.py`) with a `/predict` endpoint to accept customer data and return churn predictions.
2. **Model Serialization**: Saved the trained model using `joblib` for easy loading in the Flask application.

## Project Structure

The project is organized as follows:

Customer-Churn-Prediction/ │ ├── Data-understanding-preprocessing.ipynb # Jupyter Notebook containing data exploration, preprocessing, and model training steps ├── model_training.py # Python script for training and evaluating machine learning models ├── app.py # Flask API script for model deployment ├── requirements.txt # File listing the Python dependencies required for the project ├── README.md # Project documentation (this file) ├── LICENSE # License for the project └── images/ # Folder containing images and visualizations used in the project report


### Main Components
- **`Data-understanding-preprocessing.ipynb`**: This Jupyter Notebook includes data loading, data exploration (EDA), feature engineering, and the initial model training process. It provides detailed insights into the data and demonstrates the steps taken to build predictive models.
  
- **`model_training.py`**: This script contains the code for training various machine learning models and performing hyperparameter tuning. The models include Logistic Regression, Random Forest, and XGBoost, and the script evaluates each model's performance using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.

- **`app.py`**: The Flask application that serves the trained machine learning model as a RESTful API. This API allows for real-time predictions by sending HTTP POST requests with customer data.

- **`requirements.txt`**: Lists all the libraries and dependencies required to run the project, ensuring that the environment can be easily recreated.

- **`README.md`**: The comprehensive project documentation that explains each phase of the project, from data exploration to model deployment.

- **`images/`**: Contains all visualizations generated during the exploratory data analysis and any images used to explain the project in the README or other documentation.

## Getting Started

### Prerequisites
To run this project, you will need:
- **Python 3.8+** installed on your system
- Install the required libraries using:
  ```bash
  pip install -r requirements.txt



