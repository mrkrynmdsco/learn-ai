
import tensorflow as tf


with tf.Session() as sess:
    # Phase 1: constructing the graph
    a = tf.constant(15, name='a')
    b = tf.constant(5, name='b')
    _prod = tf.multiply(a, b, name='Multiply')
    _sum = tf.add(a, b, name='Add')
    _res = tf.divide(_prod, _sum, name='Divide')

    # Phase 2: running the session
    out = sess.run(_res)
    print(out)
