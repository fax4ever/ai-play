import tensorflow as tf
print(tf.__version__)

c1 = tf.constant(3)
print(c1)

print(c1 * 10 + 6)

m1 = tf.constant([[1, 0], [2, -3], [0.1, 0.2]])
v1 = tf.constant([[1], [2.0]])
print(m1 @ v1)

