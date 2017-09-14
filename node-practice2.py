import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tf warnings

import tensorflow as tf

a = tf.constant(1.2)
b = tf.constant(2.3)

c = tf.multiply(a, b)

d = tf.sin(c)

e = tf.div(b, d)

with tf.Session() as sess:
    outs = sess.run(e)
print("outs = {}".format(outs))

print(type(outs))