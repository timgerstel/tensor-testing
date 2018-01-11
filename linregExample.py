from __future__ import print_function
import tensorflow as tf

#modeling the equation y = W*x + b
# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input/output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x + b
# loss function
loss = tf.reduce_sum(tf.square(linear_model - y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# Training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# Training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
cur_w, cur_b, cur_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(cur_w, cur_b, cur_loss))