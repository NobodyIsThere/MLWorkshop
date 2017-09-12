import tensorflow as tf

hello = tf.constant('Congratulations! TensorFlow is set up and works properly.')

with tf.Session() as sess:
    print(sess.run([hello]))