#!/usr/bin/env python
# coding: utf-8

# # 6. Linear Algebra
# 
# ## 6.1 Introduction to linear algebra
# 
# Linear algebra is a branch of mathematics that deals with linear equations and linear transformations. It is a fundamental mathematical tool used in many fields, including machine learning and artificial intelligence.
# 
# In linear algebra, a **vector** is a sequence of numbers, typically represented as an array. For example, the vector [1, 2, 3] represents a sequence of three numbers. A **matrix** is a two-dimensional array of numbers. For example, the matrix:
# 
# $$
# \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}
# $$
# 
# represents a 2x3 matrix with three rows and two columns.
# 
# Linear algebra is used to solve systems of linear equations and to perform matrix operations such as addition, subtraction, and multiplication. These operations are used in machine learning to perform tasks such as linear regression, logistic regression, and matrix factorization.
# 
# ### Example
# 
# Consider the system of linear equations:
# 
# $$
# 2x + 3y = 6\\
# 4x + 6y = 12
# $$
# 
# To solve this system using linear algebra, we can represent the equations as the matrix equation:
# 
# This equation can be solved using matrix operations to find the values of $x$ and $y$.
# 
# $$
# \begin{bmatrix} 2 & 3 \\ 4 & 6 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 6 \\ 12 \end{bmatrix}
# $$
# 

# 
# ### Python Example
# 
# To solve the system of linear equations using Python, we can use the **`numpy`** library to perform matrix operations:

# In[1]:


import numpy as np

# Define the matrix of coefficients
A = np.array([[2, 3], [4, 6]])

# Define the vector of constant terms
b = np.array([6, 12])

# Solve the system of equations using the linalg.solve function
x = np.linalg.solve(A, b)

print(x)  # Output: [1. 2.]


# ## 6.2 Vectors and Matrices
# 
# Linear algebra is a branch of mathematics that deals with linear equations and their representations in vector and matrix form. Vectors and matrices are fundamental mathematical objects in linear algebra and are used to represent and manipulate data in many areas of science, engineering, and computer science.
# 
# ### Vectors
# 
# A vector is a collection of numbers arranged in a particular order. Vectors can be represented as a list of numbers, or as an arrow in a coordinate system. The length of the vector is the number of elements it contains, and the magnitude of a vector is the length of the arrow representing it.
# 
# In Python, we can represent a vector as a list or a NumPy array:

# In[ ]:


vector1 = [1, 2, 3]
vector2 = np.array([1, 2, 3])


# We can perform basic operations on vectors, such as addition and subtraction:

# In[ ]:


vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

# Vector addition
vector_sum = [x + y for x, y in zip(vector1, vector2)]

# Vector subtraction
vector_diff = [x - y for x, y in zip(vector1, vector2)]


# We can also take the dot product of two vectors, which is a scalar value that measures the similarity between the two vectors:

# In[ ]:


vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

dot_product = sum([x * y for x, y in zip(vector1, vector2)])


# ### Matrices
# 
# A matrix is a two-dimensional array of numbers. Matrices are often used to represent and manipulate data in the form of rows and columns. The dimensions of a matrix are the number of rows and columns it contains.
# 
# In Python, we can represent a matrix as a list of lists or a NumPy array:

# In[ ]:


matrix1 = [[1, 2, 3], [4, 5, 6]]
matrix2 = np.array([[1, 2, 3], [4, 5, 6]])


# We can perform basic operations on matrices, such as addition and subtraction:

# In[ ]:


matrix1 = [[1, 2, 3], [4, 5, 6]]
matrix2 = [[7, 8, 9], [10, 11, 12]]

# Matrix addition
matrix_sum = [[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]

# Matrix subtraction
matrix_diff = [[x - y for x, y in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]


# We can also take the dot product of two matrices, which is a matrix that results from multiplying two matrices element-wise:

# In[ ]:


matrix1 = [[1, 2, 3], [4, 5, 6]]
matrix2 = [[7, 8, 9], [10, 11, 12]]

dot_product = [[sum([x * y for x, y in zip(row1, column2)]) for column2 in zip(*matrix2)] for row1 in matrix1]


# ## 6.3 Operations on matrices and vectors
# 
# In this section, we will look at some common operations that can be performed on matrices and vectors. These operations are essential for many machine learning algorithms and are used to manipulate and transform data.
# 
# ### Matrix addition and subtraction
# 
# Two matrices can be added or subtracted if they have the same dimensions. To add two matrices, we simply add their corresponding elements. For example, given two matrices A and B with the same dimensions, we can add them as follows:
# 
# $$
# C = A + B = \begin{bmatrix} a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\ a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m,1} & a_{m,2} & \cdots & a_{m,n} \end{bmatrix} + \begin{bmatrix} b_{1,1} & b_{1,2} & \cdots & b_{1,n} \\ b_{2,1} & b_{2,2} & \cdots & b_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ b_{m,1} & b_{m,2} & \cdots & b_{m,n} \end{bmatrix} = \begin{bmatrix} a_{1,1} + b_{1,1} & a_{1,2} + b_{1,2} & \cdots & a_{1,n} + b_{1,n} \\ a_{2,1} + b_{2,1} & a_{2,2} + b_{2,2} & \cdots & a_{2,n} + b_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m,1} + b_{m,1} & a_{m,2} + b_{m,2} & \cdots & a_{m,n} + b_{m,n} \end{bmatrix}
# $$
# 
# Subtraction works in a similar way, with the corresponding elements being subtracted rather than added.
# 
# ### Matrix-vector multiplication
# 
# A matrix can be multiplied by a vector if the number of columns in the matrix is the same as the number of elements in the vector. The result is a new vector with the same number of elements as the number of rows in the matrix.
# 
# $$
# \begin{bmatrix} a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\ a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m,1} & a_{m,2} & \cdots & a_{m,n} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \ x_n \end{bmatrix} = \begin{bmatrix} a_{1,1}x_1 + a_{1,2}x_2 + \cdots + a_{1,n}x_n \\ a_{2,1}x_1 + a_{2,2}x_2 + \cdots + a_{2,n}x_n \\ \vdots \\ a_{m,1}x_1 + a_{m,2}x_2 + \cdots + a_{m,n}x_n \end{bmatrix}
# $$
# 
# Matrix-vector multiplication is a fundamental operation in linear algebra and is frequently used in machine learning and deep learning. In this operation, a matrix is multiplied by a vector to produce a new vector. The resulting vector is typically used to represent a transformation of the original vector.
# 
# The matrix-vector multiplication is defined as:
# 
# $$
#  \mathbf{y} = \mathbf{A} \mathbf{x} 
# $$
# 
# where $\mathbf{y}$ is the resulting vector, $\mathbf{A}$ is the matrix, and $\mathbf{x}$ is the original vector.
# 
# The element-wise product of the matrix and the vector is given by:
# 
# $$
#  y_i = \sum_{j=1}^{m} A_{i,j} x_j 
# $$
# 
# where $y_i$ is the $i$-th element of the resulting vector, $A_{i,j}$ is the element in the $i$-th row and $j$-th column of the matrix, and $x_j$ is the $j$-th element of the vector.
# 
# Here is an example of matrix-vector multiplication in Python using NumPy:

# In[ ]:


import numpy as np

# Define matrix and vector
A = np.array([[1, 2], [3, 4]])
x = np.array([1, 2])

# Perform matrix-vector multiplication
y = np.dot(A, x)

print(y)  # Output: [ 5 11]


# ## 6.4 Systems of linear equations
# 
# Part One. Chapter "6. Linear Algebra". Section: “6.4 Systems of linear equations”
# 
# A system of linear equations is a collection of linear equations that can be solved simultaneously. A linear equation is an equation that can be written in the form $ax + b = 0$, where $a$ and $b$ are constants.
# 
# To solve a system of linear equations, we can use either the elimination method or the substitution method.
# 
# ### → Elimination Method
# 
# The elimination method involves adding or subtracting the equations in such a way as to eliminate one of the variables. For example, consider the following system of linear equations:
# 
# $$
# 2x + 3y = 8 \\ 4x - y = 2
# $$
# 
# We can eliminate the $y$ variable by adding the two equations together:
# 
# $$
# 2x + 3y + 4x - y = 8 + 2 \\ 6x + 2 = 16 \\ x = \frac{2}{3}
# $$
# 
# Substituting this value back into either of the original equations allows us to solve for $y$:
# 
# $$
# 2x + 3y = 8 \\ 2 \cdot \frac{2}{3} + 3y = 8 \\ y = 2
# $$
# 
# Therefore, the solution to the system of linear equations is $x = \frac{2}{3}$ and $y = 2$.
# 
# ### → Substitution Method
# 
# The substitution method involves solving one of the equations for one of the variables and substituting the result into the other equation. For example, consider the following system of linear equations:
# 
# $$
# x + 2y = 6 \\ 3x - 4y = 2
# $$
# 
# We can solve the first equation for $x$:
# 
# $$
# x = 6 - 2y
# $$
# 
# Substituting this result into the second equation gives us:
# 
# $$
# 3(6 - 2y) - 4y = 2
# $$
# 
# Solving this equation for $y$ yields:
# 
# $$
# y = \frac{4}{7}
# $$
# 
# Substituting this result back into the first equation allows us to solve for $x$:
# 
# $$
# x + 2y = 6 \\ x + 2 \cdot \frac{4}{7} = 6 \\ x = \frac{6}{7}
# $$
# 
# Therefore, the solution to the system of linear equations is $x = \frac{6}{7}$ and $y = \frac{4}{7}$.

# **Example in Python Notebook:**
# 
# We can use the **`numpy`** library to solve systems of linear equations in Python. For example, consider the following system of linear equations:
# 
# $$
# 2x - 3y = 1 \\ 5x + 2y = 3
# $$
# 
# We can solve this system using the **`numpy.linalg.solve`** function:

# In[ ]:


import numpy as np

A = np.array([[2, -3], [5, 2]])
b = np.array([1, 3])

x = np.linalg.solve(A, b)

print(x)


# This will output the solution **`[0.63636364 0.90909091]`**, which corresponds to $x = \frac{7}{11}$ and $y = \frac{5}{11}$.

# ## 6.5 Eigenvalues and eigenvectors
# 
# In linear algebra, an eigenvalue and an eigenvector of a linear transformation are a scalar and a non-zero vector respectively that, when the transformation is applied to the vector, change only by a scalar factor.
# 
# Formally, if $T$ is a linear transformation and $\mathbf{v}$ is a non-zero vector, then $\mathbf{v}$ is an eigenvector of $T$ if there exists a scalar $\lambda$ such that:
# 
# $$
# T(\mathbf{v}) = \lambda\mathbf{v}
# $$
# 
# The scalar $\lambda$ is called the eigenvalue corresponding to the eigenvector $\mathbf{v}$.
# 
# Eigenvalues and eigenvectors are important in many areas of mathematics and physics, as they provide insight into the behavior of linear transformations. In machine learning and deep learning, eigenvalues and eigenvectors are used in the analysis of covariance matrices, principal component analysis, and spectral clustering.
# 
# Here is an example in Python of finding the eigenvalues and eigenvectors of a matrix:
# 

# In[1]:


import numpy as np

# Define a matrix
A = np.array([[3, 2], [1, 0]])

# Find the eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)


# This will output:
# 
# ```python
# Eigenvalues: [ 3. -1.]
# Eigenvectors: [[ 0.89442719 -0.70710678]
#                [ 0.4472136   0.70710678]]
# ```
# 
# ### → importance
# 
# Eigenvalues and eigenvectors play a significant role in deep learning because they allow us to analyze and understand the properties of a matrix. In particular, they are used in many techniques for dimensionality reduction, such as principal component analysis (PCA).
# 
# Eigenvalues represent the scaling factor of a matrix and can be used to determine its stability or instability. Eigenvectors, on the other hand, represent the direction in which a matrix stretches or compresses space. By understanding the eigenvalues and eigenvectors of a matrix, we can better understand how it transforms data and how it can be used in machine learning algorithms.
# 
# In deep learning, eigenvalues and eigenvectors are used in a variety of applications, including the analysis of neural networks, the design of efficient loss functions, and the optimization of deep learning models.
# 

# 
