#!/usr/bin/env python
# coding: utf-8

# # 4. Computational Complexity
# 
# ## 4.1 Introduction to computational complexity
# 
# Computational complexity is a branch of theoretical computer science that studies the amount of resources (such as time and space) required to solve computational problems. It helps us understand the limits of what computers can and cannot do, and can inform the design of algorithms and data structures.
# 
# In this chapter, we will cover the basics of computational complexity, including asymptotic notation, time complexity, and space complexity. We will also discuss some common computational problems and the algorithms used to solve them, and we will explore the trade-offs involved in choosing an algorithm for a particular problem.
# 
# Here is a simple example in Python that demonstrates the concept of time complexity. We will define a function **`fibonacci(n)`** that computes the nth Fibonacci number using a recursive algorithm:
# 

# In[1]:


def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))  # prints 55


# This function has a time complexity of O(2^n), since each recursive call results in two more calls. This means that the time required to compute the nth Fibonacci number grows exponentially with n. For larger values of n, this algorithm becomes impractical to use.
# 
# In the next section, we will introduce asymptotic notation and discuss how it is used to express the time and space complexity of algorithms.
# 

# ## 4.2 Time and space complexity
# 
# In the previous section, we introduced the concept of computational complexity and gave a simple example of a function with exponential time complexity. In this section, we will introduce asymptotic notation and discuss how it is used to express the time and space complexity of algorithms.
# 
# Asymptotic notation is a mathematical notation that allows us to express the growth of a function in terms of how it scales with the size of the input. The three most common types of asymptotic notation are big O, big omega, and big theta.
# 
# - Big O notation, denoted O(f(n)), is used to express an upper bound on the time or space complexity of an algorithm. It represents the worst-case scenario, and is used to describe the behavior of an algorithm as the input size grows without bound.
# - Big omega notation, denoted Ω(f(n)), is used to express a lower bound on the time or space complexity of an algorithm. It represents the best-case scenario, and is used to describe the behavior of an algorithm as the input size grows without bound.
# - Big theta notation, denoted Θ(f(n)), is used to express both an upper and lower bound on the time or space complexity of an algorithm. It represents the average-case scenario, and is used to describe the behavior of an algorithm as the input size grows without bound.
# 
# Here is an example in Python that demonstrates the use of asymptotic notation to express the time complexity of an algorithm. We will define a function **`sum_n(n)`** that computes the sum of the first n positive integers using a loop:
# 

# In[2]:


def sum_n(n):
    total = 0
    for i in range(n+1):
        total += i
    return total

print(sum_n(10))  # prints 55


# This function has a time complexity of O(n), since the time required to compute the sum grows linearly with n. This means that the time required to compute the sum of the first n positive integers is bounded above by a constant times n.
# 
# In the next section, we will discuss some common computational problems and the algorithms used to solve them.

# ## 4.3 Common computational problems
# 
# In this section, we will discuss some common computational problems and the algorithms used to solve them.
# 
# - Sorting: Sorting algorithms are used to arrange a list of items in a particular order, such as ascending or descending. Some common sorting algorithms include bubble sort, selection sort, insertion sort, merge sort, and quick sort. The time complexity of these algorithms ranges from O(n^2) for bubble sort and selection sort, to O(n log n) for merge sort and quick sort.
# - Searching: Searching algorithms are used to find a particular item in a list. Some common search algorithms include linear search and binary search. Linear search has a time complexity of O(n), while binary search has a time complexity of O(log n).
# - Shortest path: Shortest path algorithms are used to find the shortest path between two points in a graph. Some common shortest path algorithms include Dijkstra's algorithm and A* algorithm. The time complexity of these algorithms is typically O(n log n).
# - Knapsack problem: The knapsack problem is a problem in which a person has a limited amount of space and must choose a set of items to fill it, such that the total value of the items is maximized. The knapsack problem can be solved using dynamic programming or greedy algorithms. The time complexity of dynamic programming is O(nW), where n is the number of items and W is the capacity of the knapsack. The time complexity of greedy algorithms is typically O(n log n).
# 
# Here is an example in Python that demonstrates the use of the quick sort algorithm to sort a list of integers:

# In[3]:


def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

print(quick_sort([3, 6, 8, 10, 1, 2, 1]))  # prints [1, 1, 2, 3, 6, 8, 10]


# This function has a time complexity of O(n log n), since the time required to sort the list grows logarithmically with the size of the list.

# ## 4.4 Big O notation
# 
# The big O notation is a way to describe the complexity of an algorithm in terms of the number of steps required to execute it. It is used to compare the efficiency of different algorithms and to predict how well an algorithm will scale with larger inputs.
# 
# The big O notation is defined as follows:
# 
# $O(f(n)) = { g(n) \mid \exists c > 0, n_0 > 0, \forall n \geq n_0, 0 \leq g(n) \leq c \cdot f(n) }$
# 
# This means that if an algorithm has a time complexity of $O(n^2)$, for example, then the number of steps required to execute the algorithm grows at most quadratically with the size of the input.
# 
# Here is an example of calculating the big O complexity of an algorithm in Python:

# In[4]:


def my_algorithm(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i * j
    return total

print(my_algorithm(5)) # O(n^2)
print(my_algorithm(10)) # O(n^2)


# As you can see, the time complexity of the **`my_algorithm`** function is $O(n^2)$ because it has two nested loops that each iterate over the input **`n`**. The number of steps required to execute the function grows quadratically with the size of the input.
# 

# ## 4.5 Analyzing algorithms
# 
# In order to analyze the complexity of an algorithm, it is important to consider both the time and space complexity. Time complexity refers to the number of steps required to execute the algorithm, while space complexity refers to the amount of memory required to store the data.
# 
# There are several ways to analyze the complexity of an algorithm. One common approach is to use the big O notation, as described in the previous section. Another approach is to use the Θ notation, which is a more precise way to describe the complexity of an algorithm.
# 
# Here is an example of using the Θ notation to analyze the complexity of an algorithm:
# 
# 

# In[5]:


def my_algorithm(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i * j
    return total

print(my_algorithm(5)) # Θ(n^2)
print(my_algorithm(10)) # Θ(n^2)


# In this example, the complexity of the **`my_algorithm`** function is Θ(n^2) because it has two nested loops that each iterate over the input **`n`**. The number of steps required to execute the function grows quadratically with the size of the input.
# 
# It is also important to consider the best-case, worst-case, and average-case complexity of an algorithm. The best-case complexity refers to the minimum number of steps required to execute the algorithm, while the worst-case complexity refers to the maximum number of steps required. The average-case complexity refers to the expected number of steps required to execute the algorithm.
# 
# Here is an example of analyzing the best-case, worst-case, and average-case complexity of an algorithm:

# In[6]:


def my_algorithm(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i * j
    return total

# Best-case complexity: O(1)
# Worst-case complexity: Θ(n^2)
# Average-case complexity: Θ(n^2)


# In this example, the best-case complexity of the **`my_algorithm`** function is O(1) because the inner loop will only execute once if the input **`n`** is 1. The worst-case complexity is Θ(n^2) because the inner loop will execute **`n`** times for each iteration of the outer loop, resulting in **`n^2`** total steps. The average-case complexity is also Θ(n^2) because the number of steps required to execute the function grows quadratically with the size of the input.

# 
