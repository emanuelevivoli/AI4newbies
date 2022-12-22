#!/usr/bin/env python
# coding: utf-8

# # 2. Introduction to Programming
# 
# ## 2.1 Basics of Programming in Python
# 
# In this example, we will explore some of the basic concepts of programming in Python. These concepts include variables, data types, operators, and control structures.
# 
# 

# ## Variables
# 
# In programming, a **variable** is a named location in memory that is used to store a value. In Python, you can create a variable by simply assigning a value to a name:
# 
# 

# In[1]:


# create a variable called `x` and assign it the value 10
x = 10


# Once you have created a variable, you can access its value using its name:

# In[2]:


# print the value of the variable `x`
print(x)  # prints 10


# You can also update the value of a variable by re-assigning it a new value:

# In[3]:


# update the value of the variable `x` to 20
x = 20

# print the updated value of the variable `x`
print(x)  # prints 20


# ## Data Types
# 
# In Python, every value has a **data type** that determines how it is stored and used. Some common data types in Python include:
# 
# - **Integers**: Whole numbers, such as **`1`**, **`2`**, **`3`**, etc.
# - **Floats**: Decimal numbers, such as **`1.0`**, **`2.5`**, **`3.14`**, etc.
# - **Strings**: Text values, enclosed in quotation marks, such as **`"hello"`**, **`"world"`**, etc.
# 
# You can use the **`type`** function to check the data type of a value in Python:

# In[4]:


# check the data type of an integer
print(type(1))  # prints "<class 'int'>"

# check the data type of a float
print(type(1.0))  # prints "<class 'float'>"

# check the data type of a string
print(type("hello"))  # prints "<class 'str'>"


# ## Operators
# 
# In programming, **operators** are special symbols that perform specific operations on one or more values. Some common operators in Python include:
# 
# - **`+`**: Addition
# - **``**: Subtraction
# - **``**: Multiplication
# - **`/`**: Division
# - **`%`**: Modulus (remainder after division)
# 
# You can use operators to perform calculations and assign the result to a variable:
# 

# In[5]:


# create a variable `x` and assign it the value 10
x = 10

# create a variable `y` and assign it the value 5
y = 5

# create a variable `z` and assign it the result of `x` plus `y`
z = x + y

# print the value of `z`
print(z)  # prints 15


# ## Control Structures
# 
# In programming, **control structures** are blocks of code that allow you to control the flow of your program. Some common control structures in Python include:
# 
# ### If-else Statements
# 
# Used to execute different blocks of code based on a condition.
# 

# In[6]:


# create a variable `x` and assign it the value 10
x = 10

# check if `x` is greater than 5
if x > 5:
  # if `x` is greater than 5, execute this block of code
  print("x is greater than 5")
else:
  # if `x` is not greater than 5, execute this block of code
  print("x is not greater than 5")


# Output:
# 
# ```
# x is greater than 5
# ```
# 

# 
# ### For Loops
# 
# Used to iterate over a sequence of values.

# In[7]:


# create a list of numbers
numbers = [1, 2, 3, 4, 5]

# iterate over the list of numbers
for number in numbers:
  # for each number, execute this block of code
  print(number)


# Output:
# 
# ```
# 1
# 2
# 3
# 4
# 5
# ```

# ### While Loops
# 
# Used to repeat a block of code until a condition is met.

# In[8]:


# create a variable `x` and assign it the value 0
x = 0

# repeat this block of code as long as `x` is less than 5
while x < 5:
  # increment `x` by 1
  x += 1
  # print the value of `x`
  print(x)


# Output:
# 
# ```
# 1
# 2
# 3
# 4
# 5
# ```

# 
