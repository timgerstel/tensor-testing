import tensorflow as tf
#Test Machine does not currently have >= Compute Compatibility 3.0
cfg = tf.ConfigProto()
cfg.log_device_placement = False
cfg.gpu_options.allow_growth = True
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=cfg)
# Runs the op.
print(sess.run(c))
print(tf.nn.sigmoid(1.2, 'c'))