#!/usr/bin/env python
# coding: utf-8

# # 3. Data Structures
# 
# ## 3.1 **Introduction to Data Structures**
# 
# In programming, **data structures** are used to store and organize data. Some common data structures in Python include:
# 
# ### Lists
# 
# Lists are ordered collections of items that can be of any data type. Lists are created using square brackets **`[]`** and items are separated by commas **`,`**.
# 
# 

# In[1]:


# create a list of numbers
numbers = [1, 2, 3, 4, 5]

# create a list of strings
words = ['cat', 'dog', 'bird']

# create a mixed list
mixed = [1, 'cat', 3.14, True]


# Lists are ordered, so you can access items by their index. Indexes start at 0 and go up to the length of the list minus 1.
# 

# In[2]:


# access the first item in the list of numbers
print(numbers[0])  # prints 1

# access the last item in the list of words
print(words[-1])  # prints 'bird'

# access the third item in the mixed list
print(mixed[2])  # prints 3.14


# ### Tuples
# 
# Tuples are similar to lists, but they are immutable, meaning that they cannot be modified once created. Tuples are created using parentheses **`()`** and items are separated by commas **`,`**.

# In[3]:


# create a tuple of numbers
numbers = (1, 2, 3, 4, 5)

# create a tuple of strings
words = ('cat', 'dog', 'bird')

# create a mixed tuple
mixed = (1, 'cat', 3.14, True)


# Like lists, tuples are ordered and you can access items by their index.
# 

# In[4]:


# access the first item in the tuple of numbers
print(numbers[0])  # prints 1

# access the last item in the tuple of words
print(words[-1])  # prints 'bird'

# access the third item in the mixed tuple
print(mixed[2])  # prints 3.14


# ### Dictionaries
# 
# Dictionaries are unordered collections of key-value pairs. Dictionaries are created using curly braces **`{}`** and keys and values are separated by a colon **`:`**.

# In[5]:


# create a dictionary of numbers
numbers = {'one': 1, 'two': 2, 'three': 3}

# create a dictionary of words
words = {'cat': 'feline', 'dog': 'canine', 'bird': 'avian'}

# create a mixed dictionary
mixed = {'one': 1, 'cat': 'feline', 'pi': 3.14}


# You can access values in a dictionary using their keys.

# In[6]:


# access the value of the 'one' key in the dictionary of numbers
print(numbers['one'])  # prints 1

# access the value of the 'bird' key in the dictionary of words
print(words['bird'])  # prints 'avian'

# access the value of the 'pi' key in the mixed dictionary
print(mixed['pi']) # prints 3.14


# ### Sets
# 
# Sets are unordered collections of unique items. They are written as a list of comma-separated values between curly braces, with the values separated by commas. Sets are mutable, meaning you can change their contents after they have been created.

# In[7]:


# create a set
numbers = {1, 2, 3, 4, 5}

# add an element to a set
numbers.add(6)
print(numbers)  # prints {1, 2, 3, 4, 5, 6}

# remove an element from a set
numbers.remove(3)
print(numbers)  # prints {1, 2, 4, 5, 6}

# check if an element is in a set
if 4 in numbers:
    print('4 is in the numbers set')

# get the length of a set
print(len(numbers))  # prints 5

# iterate over a set
for number in numbers:
    print(number)  # prints 1, 2, 4, 5, 6


# ## 3.2 Advanced data structures
# 
# In this section, we will cover three advanced data structures: linked lists, trees, and graphs. These data structures are useful for storing and manipulating data in more complex ways than simple data structures like lists, dictionaries, and sets.
# 
# ### 3.2.1 Linked Lists
# 
# A linked list is a linear data structure that consists of a chain of nodes, where each node contains a value and a reference to the next node in the chain. Linked lists are useful for storing data that needs to be inserted or deleted frequently, as they allow you to easily add or remove elements at the beginning or end of the list.
# 
# Here is an example of a linked list in Python:
# 

# In[8]:


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
            return
        current_node = self.head
        while current_node.next is not None:
            current_node = current_node.next
        current_node.next = new_node


# ### 3.2.2 Trees
# 
# A tree is a non-linear data structure that consists of nodes arranged in a hierarchical structure, with a root node at the top and child nodes branching out from it. Trees are useful for storing and manipulating hierarchical data, such as family trees or directory structures.
# 
# Here is an example of a tree in Python:

# In[9]:


class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, value):
        new_node = Node(value)
        self.children.append(new_node)

root = Node("root")
root.add_child("child 1")
root.add_child("child 2")
root.children[0].add_child("grandchild 1")
root.children[1].add_child("grandchild 2")


# ### 3.2.3 Graphs
# 
# A graph is a non-linear data structure that consists of a set of vertices (nodes) connected by edges. Graphs are useful for storing and manipulating data that has relationships between different elements, such as social networks or transportation networks.
# 
# Here is an example of a graph in Python:

# In[10]:


class Node:
    def __init__(self, value):
        self.value = value
        self.neighbors = []

class Graph:
    def __init__(self):
        self.nodes = []

    def add_node(self, value):
        new_node = Node(value)
        self.nodes.append(new_node)
        return new_node

    def add_edge(self, node1, node2):
        node1.neighbors.append(node2)
        node2.neighbors.append(node1)


# ## 3.3 Applications of data structures in AI
# 
# In artificial intelligence, data structures play a crucial role in the design and implementation of algorithms. For example, a tree data structure can be used to implement a decision tree algorithm, which is commonly used in supervised learning tasks. Similarly, a graph data structure can be used to represent relationships between data points in a social network, which can then be analyzed using graph theory algorithms.
# 
# One common application of data structures in AI is in the representation of state spaces in search algorithms. For example, in a search for the shortest path between two points on a map, a graph data structure can be used to represent the map, with nodes representing intersections and edges representing roads between intersections. The search algorithm can then explore the graph, using various search strategies such as breadth-first search or depth-first search, to find the shortest path.
# 
# In addition to their use in specific algorithms, data structures are also important for the efficient storage and manipulation of large datasets. For example, using an appropriate data structure can allow for fast insertion, deletion, and access of data points, which can greatly improve the speed and scalability of AI systems.
# 
# Overall, understanding and utilizing appropriate data structures is essential for the design and implementation of efficient and effective AI systems.

# 
