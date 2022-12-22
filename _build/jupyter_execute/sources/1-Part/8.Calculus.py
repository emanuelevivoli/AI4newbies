#!/usr/bin/env python
# coding: utf-8

# # 8. Numerical Calculus

# 
# 
# ## 8.1 Introduction to Calculus
# 
# Calculus is a branch of mathematics that deals with the study of rates of change and the accumulation of quantities. It is a fundamental tool in the study of artificial intelligence, particularly in the field of machine learning, where it is used to optimize models and to understand the behavior of complex systems.
# 
# In this section, we will introduce the basic concepts of calculus, including limits, derivatives, and integrals. We will also discuss how these concepts can be applied in the context of machine learning and artificial intelligence.
# 
# 

# ## 8.2 Differentiation

# 
# 
# Differentiation is a mathematical operation that involves finding the rate of change of a function at a particular point. It is an essential tool in calculus and is widely used in the field of artificial intelligence, particularly in deep learning.
# 
# To find the derivative of a function, we use the notation $f'(x)$, which is read as "the derivative of $f$ at $x$". The derivative of a function can be thought of as the slope of the function at a particular point, or the slope of the tangent line to the function at that point.
# 
# For example, consider the function $f(x) = x^2$. To find the derivative of this function at a point $x$, we can use the following formula:
# 
# $$
# f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
# $$
# 
# Plugging in the values for $f(x) = x^2$, we get:
# 
# $$
# f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} = \lim_{h \to 0} \frac{2xh + h^2}{h} = \lim_{h \to 0} (2x + h) = 2x
# $$
# 
# So the derivative of $f(x) = x^2$ at a point $x$ is $2x$.
# 
# The derivative of a function can also be used to find the maximum and minimum values of the function, as well as to solve optimization problems.
# 
# In Python, we can use the **`sympy`** library to find the derivative of a function. For example:
# 
# 

# In[1]:


from sympy import Symbol, diff

x = Symbol('x')
f = x**2

derivative = diff(f, x)
print(derivative) # prints 2*x


# ## 8.3 Integration
# 

# 
# Integration is a mathematical operation that involves calculating the area under a curve. It is the inverse operation of differentiation, which is the process of finding the slope of a curve at a particular point.
# 
# In deep learning, integration is used to compute the loss function. The loss function measures how well the model is able to predict the output given the input. It is an essential part of the training process, as it allows us to determine the accuracy of the model and make necessary adjustments to improve its performance.
# 
# There are two types of integration: definite and indefinite. Definite integration involves calculating the area between two points on a curve, while indefinite integration involves finding the antiderivative of a function, which is a function that, when differentiated, yields the original function.
# 
# 

# ### → integration and loss function
# 
# Loss functions are used in machine learning and deep learning to measure the difference between the predicted output and the true output of a model. This difference is used to update the model's parameters during training.
# 
# One common way to optimize the model's parameters is to minimize the loss function using an optimization algorithm, such as gradient descent. The goal of gradient descent is to find the values of the model's parameters that minimize the loss function. To do this, the algorithm needs to calculate the derivative of the loss function with respect to the model's parameters.
# 
# The derivative is a measure of how much the loss function changes as the model's parameters change. It can be calculated using the rules of calculus, specifically the concept of differentiation. Differentiation is a fundamental operation in calculus that allows us to find the derivative of a function.
# 
# Integration is the inverse operation of differentiation. It allows us to find the area under a curve or the integral of a function. In machine learning and deep learning, integration is often used to calculate the expected value of a function, which is a key concept in probability theory.
# 
# Together, differentiation and integration are important tools in machine learning and deep learning for optimizing models and making predictions. They are used to calculate gradients, which are essential for training models using optimization algorithms, and to calculate expected values, which are important for making probabilistic predictions.
# 
# 

# ### → example
# 
# In deep learning, we often use indefinite integration to compute the loss function, which is defined as:
# 
# $$
#  L = \frac{1}{N} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2 
# $$
# 
# Where $N$ is the number of samples, $y_i$ is the true value of the $i$-th sample, and $\hat{y}_i$ is the predicted value. The loss function is minimized during training to improve the model's prediction accuracy.
# 
# Here is an example of how to compute the loss function using indefinite integration in Python:
# 
# 

# In[ ]:


import numpy as np

# true values
y = [0, 1, 0, 1, 0, 1]

# predicted values
y_hat = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]

# compute loss
loss = np.mean((y - y_hat)**2)

print(loss)


# The output of this code should be 0.02, which is the value of the loss function.

# 
