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

# This is the main part. train the model for some given steps, using while_loop
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
            train = optimizer.apply_gradients(grads_and_vars)  # optimizer.minimize does not work !
            operations.append(train)
        with tf.control_dependencies(operations):  # first do training for this batch and then go for the next batch
            step = step + 1
        return step, total_loss, total_accuracy, total_size

    _, total_loss, total_accuracy, _ = tf.while_loop(_cond, _body, [step, total_loss, total_accuracy, total_size])
    return total_loss, total_accuracy

data = np.load("../data/mnist/mnist.npz")
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

# constructing graph for learning train_steps training and validation_steps
# training for each time of running session
train_loss, train_accuracy = train_some_steps(model=model, iterator=training_iterator,
                                              num_steps=train_steps, training=True, optimizer=optimizer)
validation_loss, validation_accuracy = train_some_steps(model=model, iterator=validation_iterator,
                                                        num_steps=validation_steps, training=False)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

epochs = 400
sess.run([training_iterator.initializer, validation_iterator.initializer])
start_time = time.time()
for epoch in range(epochs):
    _train_loss, _train_accuracy = sess.run([train_loss, train_accuracy])
    _validation_loss, _validation_accuracy = sess.run([validation_loss, validation_accuracy])
    print("epoch: " + str(epoch + 1) + "\n")
    print("train loss: " + str(_train_loss))
    print("train accuracy: " + str(_train_accuracy))
    print("\n")
    print("validation loss: " + str(_validation_loss))
    print("validation accuracy: " + str(_validation_accuracy))
    print("\n\n")
end_time = time.time()
print((end_time - start_time) / 60)




