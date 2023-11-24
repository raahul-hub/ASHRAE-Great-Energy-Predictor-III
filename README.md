# ASHRAE-Great-Energy-Predictor-III
# Overview

## Introduction

Global warming is not an issue anymore, rather it’s a situation. Since the buildings plays key role in energy utilization and CO2 emissions, bringing efficiency in Energy Consumption is the most practical way to counter the climate change. Various initiatives such as Pay -For- Performance are taken in order to encourage building owners and investors to invest in energy efficient buildings by metering energy savings and getting return-of-investment based on proven and measured savings in the buildings (performance). With the goal of attracting large scale investors and financial institutions for enabling progress in building efficiencies, American Society of Heating and Air-Conditioning Engineers (ASHRAE), founded in 1894, launched the Kaggle competition to build the machine learning model which can estimate the energy consumptions of the buildings based on their historical data.

## Business Problem

To implement the aforementioned scheme i.e. Pay -For- Performance, engineering models are required to build which can accurately predict the future energy consumption based on the historical data. The total energy saving is calculated as a difference between predicted and actual energy consumptions after retrofit.

The predicted value would indicate how much energy would have been consumed by any particular building without implementing energy efficiency measures and facilitate the process of computing how much energy has been saved after implementing energy efficiency measures.

## ML Formulation of Business Problem

ASHRAE has provided metered energy consumption data four different types of meters i.e. Chilled Water, Electric, Hot Water and Steam meters along with the building and weather data. The data is from 16 unique sites and over 1000 buildings over the period of three years.
The main problem we are solving is to predict energy meter readings based on the given data. So, the target variable is meter readings. Since the meter readings can take up any real value i.e. Continuous Dependent Variables, we will treat this problem as a regression problem.

# First Cut Approach

After gathering all the relevant data for this case study, the first thing I would like to work on is to understand the data and all the features thoroughly. This includes EDA, data cleaning and pre-processing. I will look for any anomalies in the given data which includes duplicates datapoints, missing values, data unit conversion etc. EDA will help to understand the distribution of features across the dataset, their relationship with the target variable and outlier detection. Data cleaning includes outlier treatment, missing data imputation, dropping irrelevant features, data unit conversion if required, data scaling etc.

After data pre-processing, the next step will be feature selection and feature engineering. In order to determine which feature is important, I would like to use techniques like correlation graph and Shapley value. We have weather data and timestamps are available in the data. We know that the weather has a significant impact on energy utilization and there are many new features we can compute using the weather data such as humidity.

Data splitting: As the timestamp information is available, time-based splitting is preferable method, however in order to capture all the energy utilization pattern, I will include every three-month data in a training dataset followed by the next month's data as a test dataset.

Model selection: In order to set benchmarking performance, I will train any simple model such as linear regression as a baseline model. Many case studies indicate that models such as K-NN and SVM with RBF kernels can be used as a predictive model with significant accuracy. The decision tree-based algorithms such as decision tree, RF, GBDT, CatBoost and LightGBM performs very well in predictive task. Also, deep learning algorithms such as ANN can be used as a predictive model. So, I will try all these algorithms to solve our problem statement.

Model Evaluation: There are many evaluation metrics that can be used to evaluate the performance of the regression model. ASHRAE guidelines suggest that CV(RSME) can be used in the energy prediction problems. Also, in the Kaggle competition, prescribed evaluation metrics was RMSLE. Also, RMSE is also a very good option as it indicates how large the residuals being dispersed. So, a combination of all these metrics might be helpful to evaluate our models.

Amongst all the models mentioned above, Hyperparameter tuning can be done on the best performing model, using techniques like Random Search CV and Grid Search to finetune the model further to improve the performance.

Prediction: Target values of the test data will be predicted using the fine-tuned model and will be uploaded to the Kaggle website to get the leader board score.
del further to improve the performance.
Prediction: Target values of the test data will be predicted using fine-tuned model and will be uploaded to Kaggle website to get the leader board score.

# Model Used.
# Baseline Model: Linear Regression

- Simple and interpretable.
- Good for establishing a baseline performance.

# Instance-based Models: K-NN and SVM with RBF Kernels

- K-NN: Effective for local patterns, sensitive to outliers.
- SVM with RBF Kernels: Suitable for non-linear patterns, robust against overfitting.

# Decision Tree-based Models: Decision Tree, Random Forest (RF), Gradient Boosting Decision Trees (GBDT), CatBoost, LightGBM

- Decision Tree: Simple, interpretable, prone to overfitting.
- Random Forest: Ensemble of decision trees, reduces overfitting.
- GBDT: Boosting method, builds trees sequentially, powerful.
- CatBoost: Handles categorical features well, efficient.
- LightGBM: Gradient boosting with a focus on efficiency, good for large datasets.

# Deep Learning Model: Artificial Neural Network (ANN)

- ANNs: Suitable for complex patterns, deep architectures can capture intricate relationships.
  
# Custom Stacking Regressor. 
