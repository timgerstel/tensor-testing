import tensorflow as tf
hello = tf.constant('Hello Tensor')
sess = tf.Session()
print(sess.run(hello))