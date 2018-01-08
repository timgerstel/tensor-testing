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
# The print statement above does not evaluate the nodes, but prints a summary of the nodes.  To evaluate, the computational graph must be run within a session
# 3) session - encapsulates the control and state of the TF runtime
sess = tf.Session()
print("sess.run([node1, node2]):", sess.run([node1, node2]))
# Tensor nodes can be combined with operations
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
# To see a picture of the computational graph, TensorBoard can be used
# A graph can be parameterized to accept external inputs using placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)
print(sess.run(adder_node, {a : 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
add_and_triple = adder_node * 3. #Adding another operation to the graph
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
# To make a model trainable, variables allow us to get new outputs with the same input
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
# To initialize all variables, you must call tf.global_variables_initializer()
init = tf.global_variables_initializer()
sess.run(init)
# init is a handle to the TF sub-graph that initializes all global variables, which are uninitialized until sess.run is called.
print("linear_model output:", sess.run(linear_model, {x: [1, 2, 3, 4]}))
#To evaluate the model on training data, we need a y placeholder to provide the desired values, and to write a loss function
# 4) loss function - measures how far apart the current model is from the provided data.
# training data
x_train = [1, 2, 3 , 4]
y_train = [0, -1, -2, -3]
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y) # creates a vector where each element is the corresponding example's error delta squared.
loss = tf.reduce_sum(squared_deltas) # creates a single scalar that abstracts the error of all examples
print(sess.run(loss, {x: x_train, y: y_train}))
# This should produce a loss value of 23.6.  We can manually fix the model by reassigning W and b to -1 and 1 respectively.
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: x_train, y: y_train}))
# Although correct, manually fixing the model defeats the purpose of machine learning.
# TF provides optimizers which slowly change each variable in order to minimize the loss function, the simplest being gradient descent.
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults
for i in range(1000):
	sess.run(train, {x: x_train, y: y_train})
print(sess.run([W, b]))
# cleaner printing of training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))