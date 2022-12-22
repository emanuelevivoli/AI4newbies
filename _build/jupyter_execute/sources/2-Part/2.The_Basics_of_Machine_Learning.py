#!/usr/bin/env python
# coding: utf-8

# # 2. The Basics of Machine Learning
# 
# # 2.1 **Introduction**
# 
# > Definition of machine learning: a set of techniques that allows machines to learn from data rather than being explicitly programmed
# > 
# > Types of machine learning: supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning
# >
# > Key concepts: overfitting, underfitting, and bias-variance tradeoff
# 
# 
# Machine learning is a set of techniques that allows machines to learn from data rather than being explicitly programmed. It is a key tool in the field of artificial intelligence and has a wide range of applications, including natural language processing, computer vision, and robotics.
# 
# There are four main types of machine learning: supervised learning, unsupervised learning, semi-supervised learning, and reinforcement learning.
# 
# - In supervised learning, the model is trained on labeled data (i.e. data with known input-output pairs). The goal is to learn a function that can predict the output for a new input based on the examples in the training data. Examples of supervised learning include linear regression, logistic regression, and support vector machines.
# - In unsupervised learning, the model is not given any labeled data and must discover the underlying structure in the data. The goal is to find patterns or relationships in the data that may not be immediately obvious. Examples of unsupervised learning include clustering and dimensionality reduction.
# - In semi-supervised learning, the model is given a small amount of labeled data and a large amount of unlabeled data. The goal is to learn from both the labeled and unlabeled data in order to improve the model's performance.
# - In reinforcement learning, the model learns by interacting with its environment and receiving feedback in the form of rewards or penalties. The goal is to learn a policy that maximizes the cumulative reward over time.
# 
# When building machine learning models, it is important to consider three key concepts: overfitting, underfitting, and the bias-variance tradeoff.
# 
# - Overfitting occurs when the model is too complex and fits the training data too closely, resulting in poor generalization to new data.
# - Underfitting occurs when the model is too simple and does not capture the underlying structure of the data, resulting in poor performance on the training data.
# - The bias-variance tradeoff refers to the balance between the model's ability to fit the training data well (low bias) and its ability to generalize to new data (low variance). Finding the right balance is key to building a successful machine learning model.

# # 2.2 **Supervised learning**
# 
# > Definition: a type of machine learning in which the model is trained on labeled data (i.e. data with known input-output pairs)
# >
# > Examples: linear regression, logistic regression, support vector machines, decision trees
# > 
# > Key formula: mean squared error (MSE) for evaluating models, defined as:
# >   $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2$$
# > where $y_i$ is the true target value for the $i$th example, $\hat{y_i}$ is the model's prediction for the $i$th example, and $n$ is the total number of examples.
# 
# 
# Supervised learning is a type of machine learning in which the model is trained on labeled data (i.e. data with known input-output pairs). The goal is to learn a function that can predict the output for a new input based on the examples in the training data.
# 
# Some examples of supervised learning algorithms include:
# 
# - Linear regression: Linear regression is a method for modeling the relationship between a dependent variable $y$ and one or more independent variables $x$ using a linear equation of the form:
# 
# $$
# y = \beta_0 + \beta_1x + \epsilon
# $$
# 
# where $\beta_0$ and $\beta_1$ are the model parameters (also called the intercept and slope, respectively) and $\epsilon$ is the error term. The goal of linear regression is to find the values of $\beta_0$ and $\beta_1$ that minimize the error between the model's predictions and the true values of $y$.
# 
# - Logistic regression: Logistic regression is a method for binary classification (i.e. predicting a binary outcome such as "yes" or "no"). It is similar to linear regression, but instead of predicting a continuous value, it predicts the probability that an example belongs to a particular class. The prediction is made using the following formula:
# 
# $$
# \hat{y} = \frac{1}{1 + e^{-(\beta_0 + \beta_1x)}}
# $$
# 
# where $\hat{y}$ is the predicted probability, $\beta_0$ and $\beta_1$ are the model parameters, and $x$ is the input. The model makes a prediction by thresholding the probability: if the probability is greater than 0.5, the prediction is "yes," and if it is less than 0.5, the prediction is "no."
# 
# - Support vector machines (SVMs): SVMs are a method for binary classification that seeks to find the hyperplane in a high-dimensional space that maximally separates the two classes. The algorithm works by finding the support vectors (i.e. the points closest to the hyperplane) and using them to define the hyperplane.
# - Decision trees: Decision trees are a method for classification and regression that involves building a tree-like model of decisions based on the features of the input data. At each internal node of the tree, the algorithm selects the feature that maximally splits the data, and then recursively builds the tree on the resulting subgroups. The final predictions are made based on the leaves of the tree.
# 
# When evaluating the performance of a supervised learning model, it is common to use a metric such as the mean squared error (MSE) or the mean absolute error (MAE).
# 
# The MSE is defined as:
# 
# $$
# MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
# $$
# 
# where $y_i$ is the true target value for the $i$th example, $\hat{y_i}$ is the model's prediction for the $i$th example, and $n$ is the total number of examples.
# 
# The MAE is defined as:
# 
# $$
# MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|
# $$
# 
# where $y_i$, $\hat{y_i}$, and $n$ are defined as above.
# 
# In addition to these metrics, it is often helpful to evaluate the model using a confusion matrix, which shows the number of true positive, true negative, false positive, and false negative predictions made by the model.
# 
# It is also important to consider the bias-variance tradeoff when building a supervised learning model. If the model has high bias (i.e. it is oversimplified and unable to capture the complexity of the data), it may underfit the training data and have poor generalization to new data. If the model has high variance (i.e. it is overly complex and sensitive to the noise in the training data), it may overfit the training data and also have poor generalization to new data. Finding the right balance between bias and variance is key to building a successful supervised learning model.
# 
# 

# # 2.3 **Unsupervised learning**
# 
# > Definition: a type of machine learning in which the model is not given any labeled data and must discover the underlying structure in the data
# 
# Examples: clustering, dimensionality reduction, anomaly detection
# 
# Key formula: within-cluster sum of squares (WCSS) for
# > 
# 
# Unsupervised learning is a type of machine learning in which the model is not given any labeled data and must discover the underlying structure in the data. The goal is to find patterns or relationships in the data that may not be immediately obvious.
# 
# Some examples of unsupervised learning algorithms include:
# 
# - Clustering: Clustering is a method for grouping examples into "clusters" based on their similarity. There are many different clustering algorithms, including k-means, which seeks to minimize the within-cluster sum of squares (WCSS), defined as:
#     
#     $$
#     WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
#     $$
#     
#     where $k$ is the number of clusters, $C_i$ is the set of points in the $i$th cluster, and $\mu_i$ is the mean of the points in the $i$th cluster.
#     
# - Dimensionality reduction: Dimensionality reduction is a method for reducing the number of features in a dataset while preserving as much of the original information as possible. This can be useful when working with datasets that have a large number of features, as it can reduce the computational burden of building models, improve the interpretability of the results, and mitigate the risk of overfitting.
#     
#     One common method for dimensionality reduction is principal component analysis (PCA). PCA seeks to find the "principal components" of the data, which are the directions in which the data varies the most. These directions are defined by the eigenvectors of the covariance matrix of the data, and the corresponding eigenvalues represent the amount of variance explained by each eigenvector.
#     
#     PCA can be formulated as an optimization problem, in which the goal is to find the linear combination of the original features that maximizes the variance of the resulting projections. This can be done using the following formula:
#     
#     $$
#     Z = XW
#     $$
#     
#     where $X$ is the original data matrix, $W$ is the matrix of weights (i.e. the eigenvectors of the covariance matrix of $X$), and $Z$ is the matrix of projections onto the principal components.
#     
#     PCA is an unsupervised method, as it does not require any labeled data. It is commonly used as a preprocessing step before building a supervised model, as it can help to reduce the dimensionality of the data and improve the model
#     
# - Anomaly detection: Anomaly detection is a method for identifying unusual or unexpected patterns in the data. It is often used in situations where it is important to detect unusual events or observations, such as fraud detection or intrusion detection. There are many different methods for detecting anomalies, including density-based methods (which identify points that are far from the densest regions of the data) and distance-based methods (which identify points that are far from the majority of other points).
# 
# Evaluating the performance of an unsupervised learning model can be challenging, as there are no known labels to compare the model's predictions to. One common approach is to use an external measure of the "quality" of the clusters, such as the WCSS for k-means clustering. Another approach is to use visualizations to examine the structure of the data and the clusters that the model has discovered.
# 
# It is also important to consider the bias-variance tradeoff when building an unsupervised learning model. If the model has high bias (i.e. it is oversimplified and unable to capture the complexity of the data), it may fail to discover important patterns in the data. If the model has high variance (i.e. it is overly sensitive to the noise in the data), it may discover patterns that are not representative of the underlying structure of the data. Finding the right balance between bias and variance is key to building a successful unsupervised learning model.
# 
# Additionally, it is often helpful to use visualization techniques (such as scatterplots or heatmaps) to examine the structure of the data and the patterns that the model has discovered. This can provide insight into the strengths and limitations of the model and can help inform the selection of appropriate parameters or algorithms.

# 
