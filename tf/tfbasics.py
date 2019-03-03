import tensorflow as tf



x = tf.constant(5)
y = tf.constant(9)
z = tf.add(x,y)
result = tf.mul(x,y)

with tf.Session() as sess:
        sess.run()
    sess.run(print(result))

