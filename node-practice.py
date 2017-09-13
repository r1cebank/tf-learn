import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tf warnings

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

d = tf.add(a, b)
c = tf.multiply(a, b)


f = tf.add(d, c)
e = tf.subtract(d, c)

g = tf.div(f, e)

with tf.Session() as sess:
    outs = sess.run(g)
print("outs = {}".format(outs))
