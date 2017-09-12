import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

h = tf.constant("Hello")
w = tf.constant(" World!")

hw = h + w

with tf.Session() as sess:
    ans = sess.run(hw)

print ans
