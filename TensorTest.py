from __future__ import print_function
import tensorflow as tf
#https://www.tensorflow.org/get_started/get_started
### The Basics ###
# Defintions:
# 1) tensor - the central unit of data in TensorFlow.  A tensor consists of a set of primitive values shaped into an array of any number of dimensions
# 1a) rank - the rank of a tensor is its number of dimensions
# Examples #
# rank 0: 3 (a scalar with shape [])
# rank 1: [1., 2., 3.] (a vector with shape [3])
# rank 2: [[1., 2., 3.], [4., 5., 6.]] (a matrix with shape [2, 3])
# rank 3: [[[1., 2., 3.,]], [[7., 8., 9.]]] (a tensor with shape [2, 1, 3])
# 2) computational graph - a series of TF operations arranged into a graph of nodes.  Each node takes zero or more tensors as inputs and produces a tensor as an output.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) #tf.float32 implicitly typed
print(node1, node2)
#The print statement above does not evaluate the nodes, but prints a summary of the nodes.  To evaluate, the computational graph must be run within a session
# 3) session - encapsulates the control and state of the TF runtime
sess = tf.Session()
print("sess.run([node1, node2]):", sess.run([node1, node2]))
#Tensor nodes can be combined with operations
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
