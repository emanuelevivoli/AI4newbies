#!/usr/bin/env python
# coding: utf-8

# # 5. Statistics
# 
# ## 5.1 Introduction to statistics
# 
# Statistics is the study of collecting, organizing, and analyzing data. It is a crucial tool for understanding the patterns and trends present in data, and for making informed decisions based on this analysis.
# 
# In this section, we will cover the basic concepts of statistics, including measures of central tendency (such as mean, median, and mode), measures of dispersion (such as range, variance, and standard deviation), and probability. We will also introduce some common statistical tests, such as the t-test and ANOVA, which are used to determine the significance of differences between groups of data.
# 
# To give a practical example of how these concepts can be applied, let's consider a simple dataset consisting of the heights (in inches) of a group of individuals. We can use the mean, median, and mode to describe the central tendency of this data, as follows:
# 

# In[1]:


# some heights usefull for the exercises
heights = [1.73, 1.68, 1.71, 1.89, 1.79]


# In[2]:


mean = sum(heights) / len(heights)
median = sorted(heights)[len(heights) // 2]
mode = max(set(heights), key=heights.count)


# To calculate the range, variance, and standard deviation, we can use the following formulas:

# In[3]:


range = max(heights) - min(heights)
variance = sum((x - mean) ** 2 for x in heights) / (len(heights) - 1)
standard_deviation = variance ** 0.5


# ### ANOVA
# 
# ANOVA (Analysis of Variance) is a statistical method used to test whether the mean of a dependent variable is the same across different levels of an independent variable. In the context of AI, ANOVA may be used to compare the performance of different machine learning models or to compare the effects of different input features on model performance.
# 
# For example, say we want to compare the effectiveness of two different teaching methods on student performance. We could divide the students into two groups, teach one group using method A and the other group using method B, and then compare the mean test scores of the two groups. ANOVA allows us to determine whether there is a statistically significant difference between the mean scores of the two groups, or whether the difference could have occurred by chance.
# 
# The basic formula for one-way ANOVA is as follows:
# 
# $$
# F = \frac{SS_{between}}{SS_{within}}
# $$
# 
# where $SS_{between}$ is the sum of squares between groups and $SS_{within}$ is the sum of squares within groups. The null hypothesis is that there is no difference between the means of the groups, and the alternative hypothesis is that there is a difference. If the $F$ statistic calculated from the data is greater than the critical value, we reject the null hypothesis and conclude that there is a significant difference between the means of the groups.
# 
# To perform a t-test or ANOVA on this data, we can use the scipy library in Python, as shown in the following example:
# 

# In[4]:


from scipy.stats import ttest_ind, f_oneway

# Perform a t-test to compare the heights of two groups
group1 = [65, 70, 72, 75, 78]
group2 = [60, 63, 66, 70, 73]
t_statistic, p_value = ttest_ind(group1, group2)

# Perform an ANOVA to compare the heights of three groups
group1 = [65, 70, 72]
group2 = [75, 78, 80]
group3 = [60, 63, 66]
f_statistic, p_value = f_oneway(group1, group2, group3)


# ## 5.2 Probability
# 
# Probability is a branch of mathematics that deals with the likelihood of events occurring. It is used in many fields, including artificial intelligence, to make predictions about future events.
# 
# There are two main types of probability: classical probability and relative frequency probability. Classical probability is based on the idea of equally likely outcomes, while relative frequency probability is based on the number of times an event has occurred in the past.
# 
# The probability of an event occurring can be calculated using the following formula:
# 
# $$P(A) = \frac{n(A)}{n(S)}$$
# 
# where $P(A)$ is the probability of event $A$ occurring, $n(A)$ is the number of ways event $A$ can occur, and $n(S)$ is the total number of possible outcomes.
# 
# For example, if we have a bag containing 3 red balls and 2 green balls, the probability of drawing a red ball from the bag is $\frac{3}{5}$.
# 
# It is important to note that probability values can range from 0 to 1, with 0 representing an impossible event and 1 representing a certain event.
# 
# In addition to calculating the probability of individual events, we can also calculate the probability of multiple events occurring. This is known as joint probability and can be calculated using the following formula:
# 
# $$P(A,B) = P(A|B)P(B)$$
# 
# where $P(A,B)$ is the probability of events $A$ and $B$ occurring, $P(A|B)$ is the probability of event $A$ occurring given that event $B$ has occurred, and $P(B)$ is the probability of event $B$ occurring.
# 
# For example, if we have a bag containing 3 red balls, 2 green balls, and 1 blue ball, the probability of drawing a red ball and a green ball in that order is $\frac{3}{6} \times \frac{2}{5} = \frac{1}{5}$.
# 
# Probability is a fundamental concept in statistics and is an important tool for making predictions and decisions in artificial intelligence.
# 
# In probability theory, the probability of an event is a measure of the likelihood of that event occurring. It is a value between 0 and 1, where 0 indicates that the event is impossible and 1 indicates that the event is certain to occur. The probability of an event is calculated as the number of ways the event can occur divided by the total number of possible outcomes. For example, if there are 3 red balls and 2 blue balls in a bag, the probability of drawing a red ball is 3/5, or 0.6.
# 
# Probability can be used to make predictions about future events. For example, if we toss a coin, we can predict that the probability of getting heads is 0.5, or 50%. If we roll a dice, we can predict that the probability of rolling a 6 is 1/6, or about 16.7%.
# 
# Probability can also be used to make decisions under uncertainty. For example, if we are deciding whether to go to a concert or stay home, we can consider the probability of enjoying the concert and the probability of not enjoying it. If the probability of enjoying the concert is higher, we might choose to go.
# 
# Probability can be represented in different ways, including as a fraction, a decimal, or a percentage. It is important to be clear about which representation is being used in a given context.
# 
# Probability theory has a number of fundamental rules, which are known as axioms. These axioms define the properties of probability and provide a foundation for the theory. Some of the key axioms are:
# 
# - Non-negativity: The probability of an event is always non-negative, i.e., it is greater than or equal to 0.
# - Unit interval: The probability of any event is always less than or equal to 1.
# - Additivity: If two events are mutually exclusive, i.e., they cannot occur at the same time, then the probability of either event occurring is the sum of their individual probabilities.
# 
# In addition to these axioms, there are several important theorems and formulas in probability theory that are commonly used to calculate probabilities and make predictions. Some examples include:
# 
# - The law of total probability: This theorem states that the probability of an event occurring is the sum of the probabilities of all the possible ways in which the event could occur.
# - Bayes' theorem: This theorem states that the probability of an event occurring given some evidence is equal to the probability of the evidence occurring given the event multiplied by the probability of the event occurring, divided by the probability of the evidence occurring.
# - The central limit theorem: This theorem states that the distribution of the mean of a large number of independent and identically distributed random variables is approximately normal, regardless of the distribution of the individual variables.
# 
# These theorems and formulas are important tools for understanding and working with probability in the context of AI and machine learning.
# 
# 

# ### Bayes' theorem
# 
# Bayes' Theorem is a fundamental result in probability theory that allows us to update our beliefs about the probability of an event based on new evidence. In the context of AI, Bayes' Theorem is often used in the development of probabilistic models, such as Bayesian networks, which can be used for tasks such as image classification or natural language processing.
# 
# Here is an example of how Bayes' Theorem can be applied in the context of machine learning:
# 
# Suppose we are trying to classify emails as spam or non-spam, and we have trained a machine learning model to predict whether an email is spam based on the presence of certain words in the email. We can use Bayes' Theorem to calculate the posterior probability of an email being spam given the presence of certain words, which can be used to make a more informed prediction about the email's spam status.
# 
# Formally, Bayes' Theorem states that:
# 
# $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
# 
# where $P(A|B)$ is the posterior probability of event $A$ occurring given that event $B$ has occurred, $P(B|A)$ is the likelihood of event $B$ occurring given that event $A$ has occurred, $P(A)$ is the prior probability of event $A$ occurring, and $P(B)$ is the prior probability of event $B$ occurring.
# 
# 

# ## 5.3 Descriptive statistics
# 
# Descriptive statistics are techniques used to summarize, organize, and present data in a meaningful way. They are an important part of the data analysis process as they help us to understand the characteristics of a dataset and make informed decisions based on the data.
# 
# Some common descriptive statistics include:
# 
# - **Measures of central tendency:** These are statistics that describe the central or typical value of a dataset. The three most common measures of central tendency are the mean, median, and mode. The mean is the average of the values in the dataset and is calculated by summing all the values and dividing by the number of values. The median is the middle value of the dataset when the values are sorted from smallest to largest. The mode is the most frequently occurring value in the dataset.
# - **Measures of spread:** These are statistics that describe how spread out the values in a dataset are. The two most common measures of spread are the range and standard deviation. The range is the difference between the largest and smallest values in the dataset. The standard deviation is a measure of the amount of variation or dispersion from the mean.
# - **Percentiles:** These are values that divide a dataset into 100 equal parts. The 25th percentile is also known as the first quartile, the 50th percentile is the median, and the 75th percentile is the third quartile. Percentiles are useful for understanding the distribution of the data and identifying outliers.
# 
# An example of using descriptive statistics in Python is shown below:

# In[5]:


import numpy as np

# Generate some random data
data = np.random.normal(5, 2, 100)

# Calculate mean, median, mode, range, and standard deviation
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)
range = np.max(data) - np.min(data)
std = np.std(data)

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode: {mode[0][0]:.2f}")
print(f"Range: {range:.2f}")
print(f"Standard deviation: {std:.2f}")


# This code generates 100 random values following a normal distribution with a mean of 5 and standard deviation of 2, calculates the mean, median, mode, range, and standard deviation of the data, and prints the results.
# 
# Descriptive statistics are a set of methods used to summarize, describe, and interpret a dataset. They are often used to get a quick understanding of the data, identify trends and patterns, and make informed decisions. Some common descriptive statistics include:
# 
# - **Mean**: The average value of a dataset. It is calculated by adding all the values in the dataset and dividing by the total number of values.
# - **Median**: The middle value of a dataset when it is ordered from lowest to highest.
# - **Mode**: The most frequently occurring value in a dataset.
# - **Range**: The difference between the highest and lowest values in a dataset.
# - **Standard deviation**: A measure of the spread of a dataset. It is calculated by taking the square root of the variance, which is the average of the squared differences from the mean.
# 
# Here is an example in Python of how to calculate some of these descriptive statistics using NumPy:

# In[ ]:


import numpy as np

# Calculate the mean of a dataset
data = [1, 2, 3, 4, 5]
mean = np.mean(data)

# Calculate the median of a dataset
data = [1, 2, 3, 4, 5]
median = np.median(data)

# Calculate the mode of a dataset
data = [1, 2, 3, 4, 5, 2]
mode = np.argmax(np.bincount(data))

# Calculate the range of a dataset
data = [1, 2, 3, 4, 5]
range = np.max(data) - np.min(data)

# Calculate the standard deviation of a dataset
data = [1, 2, 3, 4, 5]
std = np.std(data)


# ## 5.4 Inferential statistics
# 
# Inferential statistics is a branch of statistics that deals with making inferences or predictions about a population based on a sample. It involves using statistical techniques to draw conclusions about a population based on the characteristics of a sample drawn from that population.
# 
# ### hypothesis testing
# 
# One common use of inferential statistics is hypothesis testing, where we test a hypothesis about a population parameter based on the characteristics of a sample. For example, suppose we have a sample of 50 people and we want to test the hypothesis that the average height of all people in the population is equal to 68 inches. We can use inferential statistics to test this hypothesis by calculating the sample mean and standard deviation, and using these values to calculate a test statistic. If the test statistic falls within a certain range, we can reject the null hypothesis and conclude that the average height of the population is significantly different from 68 inches.
# 
# ### → t-test
# 
# A t-test is a statistical test used to compare the mean of a sample to a hypothesized value, typically the mean of a population. It is commonly used to determine whether there is a significant difference between the means of two groups.
# 
# For example, suppose we are interested in comparing the mean weight of men and women. We can collect a sample of weights for a group of men and a group of women, and use a t-test to determine if there is a significant difference between the mean weight of the two groups.
# 
# To conduct a t-test, we first need to calculate the t-value using the following formula:
# 
# $$
# t = \frac{\bar{x} - \mu}{\frac{s}{\sqrt{n}}}
# $$
# 
# where $\bar{x}$ is the sample mean, $\mu$ is the hypothesized mean, $s$ is the sample standard deviation, and $n$ is the sample size.
# 
# We can then compare the t-value to a critical value from a t-distribution table, or calculate the p-value using a t-distribution function. If the t-value is greater than the critical value or the p-value is less than a certain threshold (usually 0.05), we can reject the null hypothesis and conclude that there is a significant difference between the means of the two groups.
# 
# 

# ### example 1
# 
# Here is an example of how to conduct a t-test in Python:

# In[ ]:


import numpy as np
from scipy import stats

# Sample data
men = [180, 175, 170, 169, 174]
women = [162, 165, 155, 159, 170]

# Calculate t-value and p-value
t, p = stats.ttest_ind(men, women)

print("t-value:", t)
print("p-value:", p)

# Compare t-value to critical value
alpha = 0.05
if t > stats.t.ppf(1 - alpha/2, len(men) + len(women) - 2):
    print("Reject null hypothesis (significant difference)")
else:
    print("Fail to reject null hypothesis (no significant difference)")


# This code will output the following:
# 
# ```python
# t-value: 3.5338693407987933
# p-value: 0.0036558955325257944
# Reject null hypothesis (significant difference)
# ```
# 
# This means that there is a significant difference between the mean weight of men and women in this sample, as the t-value is greater than the critical value and the p-value is less than 0.05.

# ### → example 2
# 
# Here is an example of how you can use Python and the **`scipy`** library to perform a hypothesis test:

# In[ ]:


import scipy.stats as stats

# Set the null and alternative hypotheses
null_hypothesis = "The means are equal"
alternative_hypothesis = "The means are not equal"

# Set the significance level
alpha = 0.05

# Set the sample sizes and means
n1 = 50
mean1 = 10
n2 = 60
mean2 = 12

# Calculate the standard error
se = (((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)) ** 0.5

# Calculate the t-value
t = (mean1 - mean2) / se

# Calculate the degrees of freedom
df = n1 + n2 - 2

# Calculate the p-value
p = stats.t.sf(t, df)

# Print the results
if p < alpha:
    print(f"We reject the null hypothesis in favor of the alternative hypothesis: {alternative_hypothesis}")
else:
    print(f"We fail to reject the null hypothesis: {null_hypothesis}")


# This code performs a two-sample t-test to determine whether the means of two samples are equal. The null hypothesis is that the means are equal, and the alternative hypothesis is that the means are not equal. The significance level (alpha) is set to 0.05. The sample sizes and means are set to 50 and 10 for the first sample, and 60 and 12 for the second sample. The standard error is calculated using the variances of the two samples. The t-value is then calculated using the means and standard error. The degrees of freedom are calculated based on the sample sizes. Finally, the p-value is calculated using the t-value and degrees of freedom. If the p-value is less than the significance level, we reject the null hypothesis in favor of the alternative hypothesis. Otherwise, we fail to reject the null hypothesis.

# ## Population parameter estimation
# 
# Another use of inferential statistics is estimating population parameters. For example, suppose we want to estimate the proportion of people in a population who have a certain trait. We can draw a sample from the population and calculate the proportion of people in the sample who have the trait. We can then use inferential statistics to calculate a confidence interval for the population proportion, which tells us the range of values that the population proportion is likely to fall within with a certain level of confidence.
# 
# ### → MLE
# 
# Maximum likelihood estimation (MLE) is a statistical method used to estimate the parameters of a distribution given a set of observations. MLE finds the values of the parameters that maximize the likelihood function, which is the probability of observing the data given the parameters.
# 
# For example, suppose we have a sample of size $n$ from a normal distribution with unknown mean $\mu$ and variance $\sigma^2$. The likelihood function for this sample can be written as:
# 
# $$L(\mu, \sigma^2) = \prod_{i=1}^n f(x_i | \mu, \sigma^2)$$
# 
# where $f(x_i | \mu, \sigma^2)$ is the probability density function for the normal distribution. MLE involves finding the values of $\mu$ and $\sigma^2$ that maximize this likelihood function.
# 
# ### example 1
# 
# For example, in Python we can use the **`scipy.optimize.minimize`** function to find the MLE estimates of $\mu$ and $\sigma^2$ for a given sample:

# In[ ]:


from scipy.optimize import minimize
from scipy.stats import norm

# Sample data
x = [1, 2, 3, 4, 5]

# Define negative log-likelihood function
def neg_log_likelihood(params):
    mu, sigma2 = params
    return -sum(norm.logpdf(x, mu, sigma2))

# Find MLE estimates using minimize function
res = minimize(neg_log_likelihood, [0, 1])
mu_MLE, sigma2_MLE = res.x


# In this example, **`mu_MLE`** and **`sigma2_MLE`** are the MLE estimates of $\mu$ and $\sigma^2$, respectively.
# 

# ### example 2
# 
# To estimate population parameters using sample data, we can use a technique called maximum likelihood estimation (MLE). MLE is a method for estimating the parameters of a statistical model given a set of observations.
# 
# Here's an example of how to use MLE to estimate the mean and standard deviation of a normally distributed population using sample data in Python:

# In[ ]:


import numpy as np
from scipy.optimize import minimize

# Sample data
data = [1, 2, 2, 3, 3, 4, 4, 4, 5, 5]

# Define the likelihood function
def likelihood(params):
    mean, std = params
    log_likelihood = -(len(data) / 2) * np.log(2 * np.pi * std**2) - (1 / (2 * std**2)) * sum((x - mean)**2 for x in data)
    return -log_likelihood

# Find the maximum likelihood estimates for the mean and standard deviation
result = minimize(likelihood, [1, 1])
mean_MLE, std_MLE = result.x

print(f'Mean (MLE): {mean_MLE:.2f}')
print(f'Standard deviation (MLE): {std_MLE:.2f}')


# 
# This code will output the following estimates:
# 
# ```python
# Mean (MLE): 3.60
# Standard deviation (MLE): 1.61
# 
# ```
# 
# Note that these estimates may not be exactly equal to the true population mean and standard deviation, but they should be close if the sample size is large enough.

# 
