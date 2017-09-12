import numpy as np
import pickle
import random

# Constants
TRAIN = 0
VAL = 1
TEST = 2

class DataProvider:
    """
    Datasets stored in 2D arrays. Each row is an example.
    Assumes (for purposes of LSTM input) that the data aren't shuffled: each
    round must be a consecutive set of tricks in the data.
    """
    dataset = None
    train_set = None
    val_set = None
    test_set = None
    train_index = 0
    val_index = 0
    test_index = 0
    train_proportion = 0
    val_proportion = 0
    cross_validation = 0
    fold = 0
    input_length = 0
    target_length = 0
    num_examples = 0
    num_train_examples = 0
    num_val_examples = 0
    num_test_examples = 0
    inputs = []
    targets = []
    
    """
    N.B. if cross_validation != 0 then val_proportion is set to 0.
    """
    def __init__(self, data_file,
                 inputs, targets,
                 train_proportion=0.8,
                 val_proportion=0.1,
                 cross_validation=0):
        self.inputs = inputs
        self.targets = targets
        self.train_proportion = train_proportion
        self.val_proportion = val_proportion
        self.cross_validation = cross_validation
        
        # Load data set
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Work out splits
        if self.cross_validation != 0:
            self.train_proportion += self.val_proportion
            self.val_proportion = 0
        self.num_examples = len(data)
        self.num_train_examples = int(self.train_proportion*self.num_examples)
        self.num_val_examples = int(self.val_proportion*self.num_examples)
        self.num_test_examples = self.num_examples - self.num_train_examples - \
                                 self.num_val_examples
        
        # Work out the input and output size
        for label in inputs:
            self.input_length += len(data[0][label])
        for label in targets:
            self.target_length += len(data[0][label])
        
        # Now, create and fill full dataset
        full_set = self.generate_dataset(data)
        
        # Now split into training, validation and test sets. Remember that
        # if we're doing cross-validation, the validation set size is set to
        # 0 (we use the training set for both training and validation).
        # The test set is left out and we lose all reference to it.
        self.train_set = full_set[:self.num_train_examples]
        self.val_set = \
            full_set[self.num_train_examples:self.num_train_examples
                    + self.num_val_examples]
        self.test_set = full_set[-self.num_test_examples:]
        
        # Now shuffle each set
        random.shuffle(self.train_set)
        random.shuffle(self.val_set)
        random.shuffle(self.test_set)
        
        # If we're doing cross-validation, store the data at this point so
        # that we can make sure that each item gets to be part of each set.
        if self.cross_validation != 0:
            self.dataset = self.train_set
            self.fold_size = int(len(self.dataset)/self.cross_validation)
            self.val_set = self.dataset[:self.fold_size]
            self.train_set = self.dataset[self.fold_size:]
            self.num_train_examples = len(self.train_set)
            self.num_val_examples = len(self.val_set)
    
    def next_fold(self):
        """
        Call after each fold in cross-validation.
        """
        self.fold = (self.fold + 1) % self.cross_validation
        self.val_set = self.dataset[self.fold*self.fold_size:
                                    self.fold*self.fold_size+self.fold_size]
        self.train_set = self.dataset[:self.fold*self.fold_size] + \
                         self.dataset[self.fold*self.fold_size+self.fold_size:]
        random.shuffle(self.train_set)
        random.shuffle(self.val_set)
    
    def generate_dataset(self, data):
        # Dataset represented as a list of (list, list) tuples
        # representing (inputs, targets)
        dataset_inputs = np.zeros((len(data), self.input_length))
        dataset_targets = np.zeros((len(data), self.target_length))
        current_example = np.zeros((self.input_length))
        current_targets = np.zeros((self.target_length))
        
        for i, datum in enumerate(data):
            offset = 0 # We insert data for each label individually; this keeps
                       # track of where to put the next label's data.
            for label in self.inputs:
                d = datum[label]
                current_example[offset:offset+len(d)] = d
                offset += len(d)
            dataset_inputs[i] = current_example.copy()
            offset = 0
            for label in self.targets:
                d = datum[label]
                current_targets[offset:offset+len(d)] = d
                offset += len(d)
            dataset_targets[i] = current_targets.copy()
        return list(zip(dataset_inputs, dataset_targets))
    
    def get_data(self, dataset, start_index, num_rows):
        data = dataset[start_index:start_index+num_rows]
        x, y = zip(*data)
        return x, y
    
    def next_batch(self, mode, batch_size):
        dataset = None
        index = 0
        if mode == TRAIN:
            dataset = self.train_set
            index = self.train_index
            self.train_index += batch_size
        if mode == VAL:
            dataset = self.val_set
            index = self.val_index
            self.val_index += batch_size
        if mode == TEST:
            dataset = self.test_set
            index = self.test_index
            self.test_index += batch_size
        
        inputs, targets = self.get_data(dataset, index, batch_size)
        return inputs, targets
    
    def full_input_length(self):
        return self.input_length
    
    def num_batches(self, mode, batch_size):
        return int(self.size(mode)/batch_size)
    
    def reset(self):
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0
    
    def size(self, mode):
        if mode == TRAIN:
            return self.num_train_examples
        if mode == VAL:
            return self.num_val_examples
        if mode == TEST:
            return self.num_test_examples