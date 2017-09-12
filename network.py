#!/usr/bin/env python

import argparse
import data_provider as dp
import datetime as dt
import numpy as np
import os
import tensorflow as tf
import time

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def str_to_list(string, cast_fn=str):
    if string == "":
        return []
    return [cast_fn(s) for s in string.split(',')]

def mean_and_stderr(values):
    mean = np.mean(values)
    stderr = np.std(values, ddof=1)/np.sqrt(len(values))
    return mean, stderr
    
def load_data_provider(args):
    """
    Either load a matching data provider if one exists, or create one.
    """
    DP = dp.DataProvider(args.data_file, args.inputs, args.targets,
                         train_proportion=args.train_proportion,
                         val_proportion=args.val_proportion)
    print("Loaded dataset.")
    return DP

def count_params():
    total = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        this_var = 1
        for dim in shape:
            this_var *= int(dim)
        total += this_var
    return total

def build_network(args):
    DP = load_data_provider(args)
    
    input_length = DP.full_input_length()
    x = tf.placeholder(tf.float32, shape=[None, input_length], name="input")
    t = tf.placeholder(tf.float32, shape=[None, DP.target_length],name="target")
    last_layer = x
    
    layer_sizes = [input_length] + args.hidden_layers + [DP.target_length]
    biases = [bias_variable([size]) for size in layer_sizes[1:]]
    weights = []
    for i, size in enumerate(layer_sizes[:-1]):
        weights.append(weight_variable([size, layer_sizes[i+1]]))
    
    for i in range(0, len(layer_sizes)-2):
        last_layer = tf.nn.relu_layer(last_layer, weights[i], biases[i])
    result = tf.nn.relu_layer(last_layer, weights[-1], biases[-1])
    
    probs = tf.nn.softmax(result, name="output")
    
    # Specify the loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=result))
    
    # Train process
    train_op = tf.train.GradientDescentOptimizer(
        args.learning_rate).minimize(cross_entropy)
    
    # Evaluation process
    correct_prediction = tf.equal(tf.argmax(t, 1), tf.argmax(result, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                              name="acc")
    
    # Summary info for TensorBoard
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', cross_entropy)
    
    # Train
    accuracies = []
    for i in range(args.cross_validation if args.cross_validation > 0 else 1):
        outputs = train_network(args, DP, x, t, train_op,
                                tracked_vars=[("accuracy", accuracy),
                                              ("loss", cross_entropy)])
        accuracies.append(outputs[0])
        if args.cross_validation > 0:
            DP.next_fold()
    
    # Then do stats
    if args.cross_validation > 0:
        acc, stderr = mean_and_stderr(accuracies)
        print(accuracies)
        print("Accuracy: {}% +/- {}%".format(acc*100, stderr*100))

def build_conv_network(args):
    """
    Simple convolutional network without pooling. Ignores most input parameters.
    """
    DP = load_data_provider(args)
    
    input_length = DP.full_input_length()
    x = tf.placeholder(tf.float32, shape=[None, input_length], name="input")
    t = tf.placeholder(tf.float32, shape=[None, DP.target_length],name="target")
    reshaped_input = tf.reshape(x, [-1, 2, 7, 6])
    last_layer = tf.transpose(reshaped_input, perm=[0,2,3,1])
    
    # Convolutional layer
    num_fmaps = 4
    W = weight_variable([4, 4, 2, num_fmaps])
    b = bias_variable([num_fmaps])
    
    h = tf.nn.relu(tf.nn.conv2d(last_layer, W, strides=[1,1,1,1],
                                padding='SAME')
                   + b)
    
    # Fully connected layer
    connected_layer_size = 200
    W = weight_variable([4*3 * num_fmaps, connected_layer_size])
    b = bias_variable([connected_layer_size])
    
    h = tf.reshape(h, [-1, 4*3 * num_fmaps])
    last_layer = tf.nn.relu(tf.matmul(h, W) + b)
    
    # Softmax layer
    W = weight_variable([connected_layer_size, DP.target_length])
    b = bias_variable([DP.target_length])
    
    last_layer = tf.matmul(last_layer, W) + b
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t,
                                                            logits=last_layer)
    train_op = tf.train.AdamOptimizer(
        args.learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(t, 1), tf.argmax(last_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    train_network(args, DP, x, t, train_op,
                  tracked_vars=[("accuracy", accuracy),
                                ("loss", cross_entropy)])

def train_network(args, data_provider,
                  input_var, target_var, train_op,
                  tracked_vars=[]):
    """
    Train the network, giving feedback on the tensors in tracked_vars.
    tracked_vars should be a list of (string, tensor) tuples, e.g.
    [("accuracy", accuracy)].
    """
    tracked_var_names, tracked_var_list = zip(*tracked_vars)
    tracked_var_names = list(tracked_var_names)
    tracked_var_list = list(tracked_var_list)

    with tf.Session() as sess:
        # Boilerplate TensorBoard code
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)
        
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(tf.global_variables())
        if args.continue_from > 0:
            saver.restore(sess, tf.train.latest_checkpoint(
                                    "{}{}".format(args.model_dir, args.name)))
        
        if args.cross_validation == 0:
            print("Number of trainable parameters: {}".format(count_params()))
            print("Beginning training.")
            print("\tBaseline classification accuracy: {:.1f}%.\n"
                  "\tTraining set size: {}\n"
                  "\tValidation set size: {}\n"
                  "\tTest set size: {}".format(
                  100./data_provider.target_length,
                  data_provider.size(dp.TRAIN),
                  data_provider.size(dp.VAL),
                  data_provider.size(dp.TEST)))
        
        # Keep track of time
        elapsed = 0.
        
        for epoch in range(args.num_epochs):
            t = time.time()
            data_provider.reset()
            
            print("Beginning epoch {}.".format(args.continue_from+epoch+1))
            if args.cross_validation == 0 and args.val_proportion > 0:
                # Validation and summary stats
                inputs, targets = data_provider.next_batch(dp.VAL,
                                                    data_provider.size(dp.VAL))
                feed = { input_var: inputs, target_var: targets }
                outputs = sess.run([summaries] + tracked_var_list,
                                   feed_dict=feed)
                summary = outputs[0]
                tracked_var_outputs = outputs[1:]
                writer.add_summary(summary, epoch+args.continue_from)
                for i, name in enumerate(tracked_var_names):
                    print("\t[VAL] {}: {}".format(name, tracked_var_outputs[i]))
            
            # Training loop
            for _ in range(data_provider.num_batches(dp.TRAIN,args.batch_size)):
                inputs, targets = data_provider.next_batch(dp.TRAIN,
                                                args.batch_size)
                feed = { input_var: inputs, target_var: targets }
                outputs = sess.run(tracked_var_list+[train_op], feed_dict=feed)
            for i, name in enumerate(tracked_var_names):
                print("\t[TRAIN] {}: {}".format(name, outputs[i]))
            
            interval = time.time() - t
            elapsed += interval
            print("\tTook {:.1f} s. Total: {:.1f} s.".format(interval, elapsed))
        
        # Save finished model and graph definition
        os.makedirs("{}{}".format(args.model_dir, args.name), exist_ok=True)
        saver.save(sess, "{}{}/model_final".format(args.model_dir, args.name))
        tf.train.write_graph(sess.graph_def,
            "{}{}".format(args.model_dir, args.name), "graph")
        
        eval_set = dp.TEST if args.cross_validation == 0 else dp.VAL
        inputs, targets = data_provider.next_batch(eval_set,
                                                data_provider.size(eval_set))
        feed = { input_var: inputs, target_var: targets }
        outputs = sess.run(tracked_var_list, feed_dict=feed)
        
        if args.cross_validation == 0:
            print ("\nFinished {}. Results:".format(args.name))
            for i, name in enumerate(tracked_var_names):
                print ("\t[TEST] {}: {}".format(name, outputs[i]))
            
            saver_def = saver.as_saver_def()
            print("\nSaver details for loading into C++:")
            print("\tmodel_path: {}graph\n"
                  "\tfilename_tensor_name: {}\n"
                  "\trestore_op_name: {}".format(args.model_dir,
                                               saver_def.filename_tensor_name,
                                               saver_def.restore_op_name))
        return outputs
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="feedforward",
                        help="Network type (feedforward or conv)")
    parser.add_argument("--hidden-layers", type=str, default="",
                        help="Number and size of hidden layers. Comma-separated"
                        " list e.g. 200,100,50.")
    parser.add_argument("--inputs", type=str,
                        help="Input features to neural network.")
    parser.add_argument("--targets", type=str,
                        help="Targets for neural network.")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--continue-from", type=int, default=0,
                        help="Continue training from last checkpoint? Specify "
                        "how many epochs we have trained for already.")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--set-size", type=str, default="0.8,0.1",
                        help="Sizes of training and validation sets, relative "
                        "to the size of the full data set. Test set size is "
                        "inferred from the other two (1-train size-val size).")
    parser.add_argument("--cross-validation", type=int, default=0,
                        help="Number of folds (k) to use for k-fold "
                        "cross-validation. Overrides --set-size[1] (val size)")
    parser.add_argument("--data-file", type=str,
                        default="./data/dataset_c4.pkl")
    parser.add_argument("--model-dir", type=str, default="./models/")
    parser.add_argument("--log-dir", type=str, default="./logs/")
    parser.add_argument("--name", type=str, default="model", help="Run name")
    
    args = parser.parse_args()
    args.inputs = str_to_list(args.inputs)
    args.targets = str_to_list(args.targets)
    args.hidden_layers = str_to_list(args.hidden_layers, cast_fn=int)
    args.set_size = str_to_list(args.set_size, cast_fn=float)
    args.train_proportion = args.set_size[0]
    args.val_proportion = args.set_size[1]
    
    if args.type == "feedforward":
        build_network(args)
    elif args.type == "conv":
        build_conv_network(args)
