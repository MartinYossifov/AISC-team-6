# Bank Marketing Campaign Analysis

## Overview
This repository contains a comprehensive analysis of a bank's marketing campaign data, sourced from the UC Irvine Machine Learning Repository. Our objective is to predict the likelihood of customers subscribing to a term deposit, which can assist the bank in optimizing its marketing strategies.

## Models and Performance
We have implemented three machine learning models: Logistic Regression, Random Forest, and XGBoost, to predict the outcomes of the marketing campaign. Each model has been rigorously trained and tested, resulting in the following accuracies:
- Logistic Regression: 88%
- Random Forest: 89%
- XGBoost: 89%

## SHAP Values Interpretation
The SHAP (SHapley Additive exPlanations) plot provides insights into the feature importance and how each feature impacts the model's prediction. From the SHAP summary plot, we observe that 'duration' of the last contact is the most significant predictor, where longer calls tend to increase the likelihood of a client subscribing to a term deposit. Features related to contact methods and housing loans also play a significant role in influencing predictions. Understanding these contributions allows us to interpret the model's decision-making process in a human-understandable form. (This applies to the SHAP of the logistic regression model).

## Prediction on New Data
We have demonstrated the model's predictive power by evaluating a new customer sample point. The sample -  'age': [70], 'job': ['admin.'], 'marital': ['married'], 'education': ['secondary'], 'default': ['no'], 'balance': [500], 
    'housing': ['yes'], 'loan': ['no'], 'contact': ['cellular'], 'duration': [800], 'campaign': [2] - represents a potential client with specific attributes such as age, job, marital status, education, and more. After preprocessing the sample to match the training data's structure, the Logistic Regression model predicted a 'yes' outcome, indicating that the individual is likely to subscribe to a term deposit.

## Conclusion
The analysis and models developed provide actionable insights into the effectiveness of the bank's marketing strategies. The high accuracy of the models indicates that they can be reliable tools for the bank to forecast campaign outcomes and tailor their marketing efforts accordingly. The SHAP plot further enhances our understanding by elucidating the relationship between the features and the model's predictions.

The prediction of new data points serves as a testament to the model's applicability in real-world scenarios, assisting the bank in making informed decisions to enhance customer engagement and the success rate of their marketing campaigns.
