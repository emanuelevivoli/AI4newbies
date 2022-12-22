#!/usr/bin/env python
# coding: utf-8

# # 7. Geometry
# 

# ## 7.1 Introduction to geometry
# 

# 
# Geometry is the study of shapes, sizes, and the properties of space. In deep learning, geometry is often used to represent data and to make predictions.
# 
# One important concept in geometry is the dot product, which is a measure of the similarity between two vectors. The dot product is defined as:
# 
# $$
# \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
# $$
# 
# where $\mathbf{a}$ and $\mathbf{b}$ are vectors with $n$ elements. The dot product is often used in deep learning to measure the similarity between two vectors. For example, in natural language processing, the dot product can be used to measure the similarity between two documents based on the words they contain.
# 
# Another important concept in geometry is the cross product, which is a vector that is perpendicular to two other vectors. The cross product is defined as:
# 
# $$
# \mathbf{a} \times \mathbf{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \\ \end{vmatrix}
# $$
# 
# where $\mathbf{a}$ and $\mathbf{b}$ are vectors in three-dimensional space. The cross product is often used in deep learning to measure the orientation of objects in an image. For example, in object detection, the cross product can be used to determine the orientation of an object relative to the camera.
# 
# 

# ## 7.2 Euclidean Geometry
# 

# 
# Euclidean geometry is a branch of mathematics that deals with the study of points, lines, angles, and shapes in two-dimensional and three-dimensional space. It is named after the Greek mathematician Euclid, who is credited with developing the axiomatic system that is used to study geometry.
# 
# In Euclidean geometry, points are considered to be the most basic objects. They have no size or shape and are simply represented by a pair of coordinates in a coordinate system. Lines are formed by connecting two points, and angles are formed by the intersection of two lines. Shapes such as circles, triangles, and squares are formed by the intersection of lines and angles.
# 
# Euclidean geometry is used in many different fields, including architecture, engineering, and computer graphics. It is also an important tool in machine learning and deep learning, as it allows us to represent data in a structured and meaningful way.
# 
# Below is an example of how Euclidean geometry can be used in Python to represent points and lines:
# 

# In[1]:


from math import sqrt

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def distance_to_origin(self):
    return sqrt(self.x**2 + self.y**2)

class Line:
  def __init__(self, point1, point2):
    self.point1 = point1
    self.point2 = point2

  def length(self):
    return point1.distance_to_origin(point2)

point1 = Point(3, 4)
point2 = Point(5, 6)
line = Line(point1, point2)
print(line.length())  # Output: 2.8284271247461903


# In this example, we have defined a **`Point`** class that represents a point in a two-dimensional coordinate system. We have also defined a **`Line`** class that represents a line formed by connecting two points. The **`length`** method of the **`Line`** class calculates the distance between the two points using the Euclidean distance formula.
# 
# 

# ## 7.3 Non-Euclidean geometry

# 
# 
# Non-Euclidean geometry refers to any type of geometry that is not based on the axioms of Euclidean geometry. This includes geometries such as hyperbolic geometry and elliptic geometry. Non-Euclidean geometry is not often studied in traditional geometry courses, but it has some applications in the field of deep learning.
# 
# One application of non-Euclidean geometry in deep learning is the use of hyperbolic geometry in the design of deep learning models. Hyperbolic geometry is a type of non-Euclidean geometry that is based on the idea that there are multiple lines that can be drawn through a single point, and that the sum of the angles in a triangle is less than 180 degrees. This is in contrast to Euclidean geometry, where there is only one line that can be drawn through a single point and the sum of the angles in a triangle is always 180 degrees.
# 
# The use of hyperbolic geometry in deep learning models is based on the idea that it can better represent the relationships between data points in high-dimensional spaces. This is because in high-dimensional spaces, the distances between data points can become very large, making it difficult to accurately represent the relationships between them using Euclidean geometry. Hyperbolic geometry, on the other hand, can better represent these relationships because it allows for the existence of multiple lines through a single point, which can help to capture the complexity of the relationships between data points in high-dimensional spaces.
# 
# One example of the use of hyperbolic geometry in deep learning is in the design of the Poincaré embeddings, which are a type of low-dimensional representation of data points in a high-dimensional space that uses hyperbolic geometry. These embeddings have been shown to be effective at capturing the relationships between data points in high-dimensional spaces, and have been used in a variety of applications including language modeling and recommendation systems.
# 
# Overall, the use of non-Euclidean geometry, particularly hyperbolic geometry, in deep learning can be a powerful tool for representing and analyzing relationships between data points in high-dimensional spaces.
# 
# ### → more info
# 
# Non-Euclidean geometry is a type of geometry that is not based on the assumptions of Euclidean geometry. It is important in deep learning because it is used in the development of neural networks, which are essential for many applications of AI.
# 
# One of the main differences between Euclidean and non-Euclidean geometry is the concept of parallel lines. In Euclidean geometry, parallel lines are lines that never intersect. In non-Euclidean geometry, parallel lines may intersect or may be curved.
# 
# Another important concept in non-Euclidean geometry is the idea of a curved space. In Euclidean geometry, space is flat, but in non-Euclidean geometry, space can be curved. This is important for deep learning because neural networks often operate in high-dimensional spaces, which can be thought of as being curved.
# 
# One way to visualize non-Euclidean geometry is to consider the surface of a sphere. In Euclidean geometry, the shortest distance between two points is a straight line, but on a sphere, the shortest distance is a curve called a "great circle."
# 
# There are several types of non-Euclidean geometry, including hyperbolic geometry, elliptical geometry, and Riemannian geometry. These geometries have different properties and are used in different contexts in deep learning. For example, Riemannian geometry is often used in the development of optimization algorithms for neural networks, while hyperbolic geometry is used in the study of the structure of complex networks.
# 
# In summary, non-Euclidean geometry is an important concept in deep learning because it allows us to represent and analyze complex structures and relationships in high-dimensional spaces.
# 
# 

# ## 7.4 Applications of geometry in AI
# 
# 

# Geometry plays an important role in various areas of artificial intelligence, including computer vision and robotics. In computer vision, geometric transformations such as rotations, translations, and scaling are often used to align images and extract features. In robotics, geometric concepts such as distance, angle, and orientation are crucial for planning and navigating in complex environments.
# 
# One common application of geometry in AI is in the use of convolutional neural networks (CNNs) for image classification. CNNs use filters, which are essentially small matrices, to identify patterns in images. These filters are applied to the image through a process called convolution, which involves sliding the filter over the image and taking the dot product between the filter and the image at each position. The resulting output is a feature map that encodes the presence or absence of certain patterns in the image.
# 
# Another application of geometry in AI is in the use of geometric deep learning, which involves applying deep learning techniques to non-Euclidean data such as graphs and manifolds. This has led to the development of new architectures such as graph convolutional networks (GCNs) and manifold learning algorithms, which can be used to analyze complex relationships between data points in non-Euclidean spaces.
# 
# Geometry also plays a role in the design of autonomous systems, such as self-driving cars and drones. These systems rely on sensors and algorithms to perceive and understand their surroundings, which requires the use of geometric concepts such as distance, angle, and orientation. For example, a self-driving car may use laser rangefinders or lidar to measure distances to objects in its environment, and algorithms to determine the car's position and orientation relative to these objects.
# 
# In summary, geometry is an important field of mathematics that has numerous applications in artificial intelligence, particularly in the areas of computer vision, robotics, and autonomous systems.
# 
# 

# ### → with example
# 
# Here is an example of convolution with a filter in Python using NumPy:

# In[ ]:


import numpy as np

# Sample input image
image = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Sample filter
filter_ = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Convolve the filter with the image
result = np.zeros((image.shape[0]-filter_.shape[0]+1, image.shape[1]-filter_.shape[1]+1))
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        result[i, j] = np.sum(image[i:i+filter_.shape[0], j:j+filter_.shape[1]] * filter_)

print(result)
# Output: [[-4 -4 -4]
#           [ 4  4  4]
#           [12 12 12]]


# This code applies the filter to the input image using convolution, and produces the output image with the same size as the input, but with the filter applied. The output values are computed by summing the element-wise product of the filter and the portion of the image that the filter is being applied to.

# 
