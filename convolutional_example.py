import data_provider as dp
import tensorflow as tf

# Parameters
num_fmaps = 4           # Number of feature maps for the convolutional layer
connected_size = 200    # Number of units in the fully connected layer
learning_rate = 1e-4    # Learning rate for the Adam optimiser
num_epochs = 10         # Number of training epochs
batch_size = 100        # Number of training examples per batch

# Load data
print("Loading data...")
DP = dp.DataProvider("./data/dataset_c4.pkl",           # Data file
                    ["own_pieces", "opponent_pieces"],  # Inputs
                    ["move"],                           # Targets
                    train_proportion=0.8,
                    val_proportion=0.1)

# Build network
input_length = DP.full_input_length()
target_length = DP.target_length
x = tf.placeholder(tf.float32, shape=[None, input_length], name="input")
t = tf.placeholder(tf.float32, shape=[None, target_length], name="target")

# Reshape input from a vector into two grids
reshaped_input = tf.reshape(x, [-1, 2, 7, 6])
input_layer = tf.transpose(reshaped_input, perm=[0, 2, 3, 1])
# The dimensions of input_layer now correspond to [batch, width, height, player]

# Convolutional layer
Wc = tf.Variable(tf.truncated_normal([4, 4, 2, num_fmaps], stddev=0.1))
bc = tf.Variable(tf.constant(0.1, shape=[num_fmaps]))

conv_layer = tf.nn.relu(tf.nn.conv2d(input_layer, Wc, strides=[1, 1, 1, 1],
                                                      padding='SAME') + bc)

# Fully connected layer
Wfc = tf.Variable(tf.truncated_normal([7*6*num_fmaps, connected_size],
                                      stddev=0.1))
bfc = tf.Variable(tf.constant(0.1, shape=[connected_size]))

flattened_convolution = tf.reshape(conv_layer, [-1, 7*6*num_fmaps])
connected_layer = tf.nn.relu_layer(flattened_convolution, Wfc, bfc)

# Output layer
Wo = tf.Variable(tf.truncated_normal([connected_size, target_length],
                                     stddev=0.1))
bo = tf.Variable(tf.constant(0.1, shape=[target_length]))

output_layer = tf.matmul(connected_layer, Wo) + bo

# Specify loss and train network as before, this time using Adam optimiser.
# (Gradient descent would also work fine.)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=output_layer))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct = tf.equal(tf.argmax(t, 1), tf.argmax(output_layer, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        DP.reset()
        
        print("Beginning epoch {}.".format(epoch+1))
        # Calculate accuracy on validation set.
        inputs, targets = DP.next_batch(dp.VAL, DP.size(dp.VAL))
        acc = sess.run(accuracy, feed_dict={ x: inputs, t: targets })
        print("Validation set accuracy: {:.1f}%.".format(acc*100))
        
        # Train on training set in batches
        for _ in range(DP.num_batches(dp.TRAIN, batch_size)):
            inputs, targets = DP.next_batch(dp.TRAIN, batch_size)
            acc = sess.run(train_op, feed_dict={ x: inputs, t: targets })
    
    print("Finished training.")
    # Calculate accuracy on test set.
    inputs, targets = DP.next_batch(dp.TEST, DP.size(dp.TEST))
    acc = sess.run(accuracy, feed_dict={x: inputs, t: targets})
    print("Test set accuracy: {:.1f}%.".format(acc*100))