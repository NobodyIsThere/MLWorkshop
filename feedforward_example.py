import data_provider as dp
import tensorflow as tf

# Parameters
hidden_size = 200       # Number of units in the hidden layer
learning_rate = 0.01    # Learning rate for gradient descent optimiser
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

# Define inputs and outputs
input_length = DP.full_input_length()
target_length = DP.target_length
x = tf.placeholder(tf.float32, shape=[None, input_length], name="input")
t = tf.placeholder(tf.float32, shape=[None, target_length], name="target")

# Weights and biases for hidden layer
Wh = tf.Variable(tf.truncated_normal([input_length, hidden_size], stddev=0.1))
bh = tf.Variable(tf.constant(0.1, shape=[hidden_size]))

# Weights and biases for the output layer
Wo = tf.Variable(tf.truncated_normal([hidden_size, target_length], stddev=0.1))
bo = tf.Variable(tf.constant(0.1, shape=[target_length]))

# Layers
hidden_layer = tf.nn.relu_layer(x, Wh, bh)
output_layer = tf.matmul(hidden_layer, Wo) + bo

# Generally, in a classification problem, we want to be able to interpret the
# outputs of the final layer as probabilities that the input belongs to each
# class. We do this when we calculate the loss, below: the
# softmax_cross_entropy_with_logits() function normalises the output layer so
# that it adds up to 1 before calculating the loss. If we just want the
# normalisation (for example, to output to our Connect Four AI), we can use
final_output = tf.nn.softmax(output_layer)
# N.B. in this code, this value is never calculated, because all we do here is
# train the network. But if we wanted to use the network after it was trained,
# this is the value we would use.
# i.e. output = sess.run(final_output, feed_dict={ x: my_input })

# We have built the network. Now specify loss function for training.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=output_layer))

# We want to minimise the loss, using gradient descent. Every time we evaluate
# train_op, the optimiser will make one step.
train_op = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy)

# Loss is not very interpretable, so calculate the accuracy.
correct = tf.equal(tf.argmax(output_layer, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Now actually train the network!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        DP.reset()
        
        print("Beginning epoch {}.".format(epoch+1))
        # Calculate accuracy on validation set.
        inputs, targets = DP.next_batch(dp.VAL, DP.size(dp.VAL))
        acc, loss = sess.run([accuracy, cross_entropy],
                             feed_dict={ x: inputs, t: targets })
        print("Validation set accuracy: {:.1f}%.".format(acc*100))
        print("Validation set loss: {:.7f}".format(loss))
        
        # Train on training set in batches
        for _ in range(DP.num_batches(dp.TRAIN, batch_size)):
            inputs, targets = DP.next_batch(dp.TRAIN, batch_size)
            acc = sess.run(train_op, feed_dict={ x: inputs, t: targets })
    
    print("Finished training.")
    # Calculate accuracy on test set.
    inputs, targets = DP.next_batch(dp.TEST, DP.size(dp.TEST))
    acc, loss = sess.run([accuracy, cross_entropy],
                         feed_dict={x: inputs, t: targets})
    print("Test set accuracy: {:.1f}%.".format(acc*100))
    print("Test set loss: {:.7f}".format(loss))
