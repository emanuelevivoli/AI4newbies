#!/usr/bin/env python
# coding: utf-8

# # 1. Fraud Detection

# ## 1.1 Introduction to fraud detection
# 
# Fraud detection is a common application of artificial intelligence in many industries. It involves using machine learning algorithms to analyze patterns in data and identify unusual or suspicious activity. Fraud can take many forms, including financial fraud, identity theft, and other types of crime. In order to detect fraud effectively, it is important to have a good understanding of the types of fraud that can occur, as well as the characteristics of fraudulent activity.
# 
# There are many different techniques that can be used for fraud detection, including supervised learning, unsupervised learning, and semi-supervised learning. In supervised learning, the machine learning algorithm is trained on a labeled dataset, which includes both fraudulent and non-fraudulent examples. The algorithm then uses this training data to learn to distinguish between fraudulent and non-fraudulent activity. In unsupervised learning, the algorithm is not given any labeled examples, but is instead expected to discover patterns in the data on its own. Semi-supervised learning combines elements of both supervised and unsupervised learning, using a small number of labeled examples to guide the algorithm in its discovery of patterns in the data.
# 
# One of the challenges of fraud detection is the imbalanced nature of the data. Fraudulent activity is typically much rarer than non-fraudulent activity, which can make it difficult for machine learning algorithms to accurately identify fraudulent patterns. To address this issue, it is often necessary to use techniques such as oversampling and undersampling to balance the data and improve the accuracy of the machine learning model.

# ## 1.2 Types of fraud
# 
# Fraud can take many forms and can occur in a variety of contexts. Some common types of fraud include:
# 
# - Financial fraud, Financial fraud involves the use of deceptive practices for financial gain. This can include things like credit card fraud, identity theft, and money laundering.
# - Insurance fraud, Insurance fraud involves making false claims or providing false information in order to obtain insurance benefits. This can include things like staging accidents, exaggerating the extent of an injury, or pretending to have lost property that was never actually lost.
# - Investment fraud, Investment fraud involves persuading individuals to invest in a scheme that is not legitimate. This can include things like Ponzi schemes, pyramid schemes, and investment fraud involving stocks or securities.
# - Health care fraud, Health care fraud involves making false claims or providing false information in order to obtain health care benefits. This can include things like billing for services that were not provided, using false diagnoses to justify unnecessary treatments, and prescribing unnecessary medications.
# 
# ### Example in Python
# 
# Here is an example of how fraud detection techniques might be applied in Python:
# 

# In[1]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the data
X = np.load("fraud_data.npy")
y = np.load("fraud_labels.npy")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print("Test set accuracy: {:.2f}".format(accuracy))


# This code uses a random forest classifier to train a model on data related to fraud, and then uses the model to predict whether a given transaction is fraudulent or not. The model is evaluated on a test set to determine its accuracy.

# ## 1.3 Detection methods
# 
# There are several methods that can be used to detect fraudulent activity. Some of the most common methods include:
# 
# - **Rule-based systems**: These systems rely on a set of pre-defined rules to identify fraudulent activity. For example, a rule-based system might flag any transactions that are larger than a certain amount as potentially fraudulent.
# - **Behavioral analysis**: This approach involves analyzing the behavior of individuals or groups in order to identify patterns that may indicate fraudulent activity. For example, if an individual's spending patterns suddenly change, this could be a sign of fraudulent activity.
# - **Machine learning**: Machine learning algorithms can be trained to identify patterns in data that may indicate fraudulent activity. For example, a machine learning model might be trained to identify patterns in transaction data that are characteristic of fraudulent activity.
# - **Anomaly detection**: This approach involves identifying deviations from normal behavior that may indicate fraudulent activity. For example, if an individual's spending patterns are significantly different from those of their peers, this could be a sign of fraudulent activity.
# 
# It's important to note that no single method is foolproof, and the most effective fraud detection systems often combine multiple approaches in order to achieve the best results.

# 
# 
# ## 1.4 Case study: Credit card fraud detection
# 
# One common application of AI in fraud detection is the detection of fraudulent credit card transactions. In this case study, we will explore a real-world example of using machine learning to detect fraudulent credit card transactions.
# 
# The dataset used in this case study is the Credit Card Fraud Detection dataset from Kaggle (**[https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)**). It contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly unbalanced, with only 0.172% of transactions being fraudulent.
# 
# We will start by exploring the data and performing some preprocessing steps. Then, we will train a machine learning model to classify transactions as either fraudulent or non-fraudulent. Finally, we will evaluate the model's performance and discuss some potential improvements.
# 
# ### 1.4.1 Exploring the data
# 
# First, let's import the necessary libraries and load the data:
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('creditcard.csv')


# Next, let's take a look at the data to get a sense of the features and their distributions:

# In[ ]:


df.head()


# This dataset contains 31 features, including the time and amount of the transaction, as well as various anonymized features. The target variable, **`Class`**, indicates whether the transaction was fraudulent (1) or non-fraudulent (0).
# 
# Now, let's plot the distribution of the target variable to get a sense of the imbalance in the data:
# 

# In[ ]:


# Count the number of fraudulent and non-fraudulent transactions
fraud_count = df[df['Class'] == 1].shape[0]
non_fraud_count = df[df['Class'] == 0].shape[0]

# Plot the distribution
plt.bar(['Fraudulent', 'Non-fraudulent'], [fraud_count, non_fraud_count])
plt.show()


# As we can see, the majority of the transactions are non-fraudulent, with only a small fraction being fraudulent. This imbalance will be important to consider when evaluating the performance of our model.

# ### 1.4.2 Preprocessing
# 
# Before training a model, we need to perform some preprocessing steps on the data. First, let's separate the features and target variable:
# 

# In[ ]:


# Separate the features and target variable
X = df.drop('Class', axis=1)
y = df['Class']


# Next, we will scale the features using the **`StandardScaler`** from scikit-learn:
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Finally, we will split the data into training and testing sets.
# 
# To split the data into training and testing sets, we can use the **`train_test_split`** function from the **`sklearn.model_selection`** module. First, we need to import the function. 
# Then, we can split the data into training and testing sets with the following code:
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Here, **`X`** is the input data and **`y`** is the target variable. The **`test_size`** parameter specifies the proportion of the data that will be used for testing. In this case, we are using 20% of the data for testing. The **`random_state`** parameter is used to set the random seed, which ensures that the data is split in the same way each time the code is run.
# 
# After splitting the data, we can use the training set to train the model and the testing set to evaluate its performance.
# 
# Finally, we will split the data into training and testing sets:
# 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# The **`train_test_split`** function from scikit-learn's **`model_selection`** module allows us to split our data into training and testing sets. The **`test_size`** parameter determines the proportion of the data that will be used for testing, and the **`random_state`** parameter ensures that the data is split in a reproducible manner.
# 
# Next, we will train a logistic regression model on the training data:
# 

# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# The **`LogisticRegression`** class from scikit-learn's **`linear_model`** module implements logistic regression. The **`fit`** method trains the model on the training data.
# 
# Once the model is trained, we can use it to make predictions on the testing data:
# 

# In[ ]:


y_pred = log_reg.predict(X_test)


# The **`predict`** method returns a list of predictions for the testing data.
# 
# Finally, we can evaluate the model's performance by computing the confusion matrix and several metrics:
# 

# In[ ]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", confusion_matrix)

precision = precision_score(y_test, y_pred)
print("Precision: {:.2f}".format(precision))

recall = recall_score(y_test, y_pred)
print("Recall: {:.2f}".format(recall))

f1 = f1_score(y_test, y_pred)
print("F1 score: {:.2f}".format(f1))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))


# The **`confusion_matrix`**, **`precision_score`**, **`recall_score`**, **`f1_score`**, and **`accuracy_score`** functions are all from scikit-learn's **`metrics`** module and allow us to compute various metrics for evaluating the model's performance. The confusion matrix allows us to see the number of true positive, true negative, false positive, and false negative predictions. Precision is the ratio of true positive predictions to all positive predictions. Recall is the ratio of true positive predictions to all actual positive samples. The F1 score is the harmonic mean of precision and recall. Accuracy is the ratio of correct predictions to all predictions.

# In[ ]:




