import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tf warnings

import tensorflow as tf
import numpy as np


c = tf.constant(4.0, dtype=tf.float64)
print(c)
print(c.dtype)

x = tf.constant([1,2,3], name='x', dtype=tf.float32)
print(x.dtype)
x = tf.cast(x,tf.int64)
print(x.dtype)

c = tf.constant([[1,2,3],
                 [4,5,6]])
print("Python List input: {}".format(c.get_shape()))

c = tf.constant(np.array([
                 [[1,2,3],
                  [4,5,6]],

                 [[1,1,1],
                  [2,2,2]]
                 ]))

print("3d NumPy array input: {}".format(c.get_shape()))

sess = tf.InteractiveSession()
c = tf.linspace(0.0, 1.0, 3)
print("The content of 'c':\n {}\n".format(c.eval()))
sess.close()

a = tf.constant([ [1,2,3],
                  [4,5,6] ])
print(a.get_shape())

x = tf.constant([1,0,1])
print(x.get_shape())

with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    c2 = tf.constant(4,dtype=tf.int32,name='c1s')
print(c1.name)
print(c2.name)

init_val = tf.random_normal((1,5),0,1)
var = tf.Variable(init_val, name='var')
print("pre run: \n{}".format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)

print("\npost run: \n{}".format(post_var))


x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32,shape=(5,10))
    w = tf.placeholder(tf.float32,shape=(10,1))
    b = tf.fill((5,1),-1.)
    xw = tf.matmul(x,w)

    xwb = xw + b
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s, feed_dict={x: x_data, w: w_data})

print("outs = {}".format(outs))

key1 = 1
key2 = 2

print ({key1: x_data, key2: w_data})