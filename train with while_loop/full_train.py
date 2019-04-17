import sys
if "../" not in sys.path:
    sys.path.append("../")
if "../utils" not in sys.path:
    sys.path.append("../utils")

import tensorflow as tf
import numpy as np
import time
from utils import update_loss, get_tensor_shape
from mnist_model import Model

# this is the main part like partial_train.py
def train_some_steps(model, iterator, num_steps, training, optimizer=None):
    if training and optimizer is None:
        raise ValueError("if training is True, then optimizer should be specified")
    step = tf.constant(0, dtype=tf.int32)
    total_loss = tf.constant(0.0, dtype=tf.float32)
    total_accuracy = tf.constant(0.0, dtype=tf.float32)
    total_size = tf.constant(0.0, dtype=tf.float32)

    def _cond(step, total_loss, total_accuracy, total_size):
        return step < num_steps

    def _body(step, total_loss, total_accuracy, total_size):
        inputs, labels = iterator.get_next()
        inputs = tf.cast(inputs, tf.float32)
        labels = tf.cast(labels, tf.int32)
        probs = model(inputs, training=training)
        labels = tf.one_hot(labels, 10)
        loss = tf.keras.losses.categorical_crossentropy(labels, probs)
        loss = tf.reduce_mean(loss)
        accuracy = tf.keras.metrics.categorical_accuracy(labels, probs)
        accuracy = tf.reduce_mean(accuracy)
        size = tf.shape(inputs)[0]
        size = tf.cast(size, tf.float32)
        total_loss = update_loss(total_loss, loss, total_size, size)
        total_accuracy = update_loss(total_accuracy, accuracy, total_size, size)
        total_size = total_size + size
        operations = [total_loss, total_accuracy, total_size]
        if training:
            grads = tf.gradients(loss, model.trainable_weights)
            grads_and_vars = zip(grads, model.trainable_weights)
            train = optimizer.apply_gradients(grads_and_vars)
            operations.append(train)
        with tf.control_dependencies(operations):  # first do training for this batch and then go for the next batch
            step = step + 1
        return step, total_loss, total_accuracy, total_size

    _, total_loss, total_accuracy, _ = tf.while_loop(_cond, _body, [step, total_loss, total_accuracy, total_size])
    return total_loss, total_accuracy

# constructing the outer for loop for training
# note the outputs will not be printed in jupyter notebook since tf.print does not work in jupyter notebook
def full_train(model, epochs, train_iterator, validation_iterator, train_steps, validation_steps, optimizer):
    epoch = tf.constant(0, dtype=tf.int32)
    def _cond(epoch):
        return epoch < epochs

    def _body(epoch):
        train_loss, train_accuracy = train_some_steps(model, train_iterator, train_steps, True, optimizer)
        operations = [train_loss, train_accuracy]
        with tf.control_dependencies(operations):
            validation_loss, validation_accuracy = train_some_steps(model, validation_iterator, validation_steps, False)
        operations = [validation_loss, validation_accuracy]
        with tf.control_dependencies(operations):
            pr = tf.print("epoch:", epoch + 1)
        with tf.control_dependencies([pr]):
            pr = tf.print("\ntrain_loss:", train_loss)
        with tf.control_dependencies([pr]):
            pr = tf.print("train_accuracy:", train_accuracy)
        with tf.control_dependencies([pr]):
            pr = tf.print("\nvalidation_loss:", validation_loss)
        with tf.control_dependencies([pr]):
            pr = tf.print("validation_accuracy:", validation_accuracy)
        with tf.control_dependencies([pr]):
            pr = tf.print("\n\n")
        with tf.control_dependencies([pr]):
            epoch = epoch + 1
        return epoch

    epoch = tf.while_loop(_cond, _body, [epoch])
    train = tf.group(epoch)
    return train


data = np.load("../data/mnist/mnist.npy")
batch_size = 256
train_data = tf.data.Dataset.from_tensor_slices((data["train_images"], data["train_labels"]))
validation_data = tf.data.Dataset.from_tensor_slices((data["test_images"], data["test_labels"]))
train_data = train_data.shuffle(60000).repeat().batch(256)
validation_data = validation_data.shuffle(10000).repeat().batch(256)
train_steps = int(60000 / batch_size + 0.5)  # steps for one epoch
validation_steps = int(10000 / batch_size + 0.5)  # steps for one epoch
data = None

training_iterator = train_data.make_initializable_iterator()
validation_iterator = validation_data.make_initializable_iterator()
model = Model(name="model")
optimizer = tf.train.AdamOptimizer(10e-5)

epochs = 400
train = full_train(model, epochs, training_iterator, validation_iterator, train_steps, validation_steps, optimizer)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

sess.run([training_iterator.initializer, validation_iterator.initializer])
start = time.time()
sess.run(train)
end = time.time()
print((end - start) / 60)
