import tensorflow as tf
hello = tf.constant('Congratulations! TensorFlow is set up and works properly.')
sess = tf.Session()
print(sess.run(hello))