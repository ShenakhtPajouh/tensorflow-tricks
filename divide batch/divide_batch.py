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

# divide batch to n sub_batches. .e.g  [256, 28, 28] -> [8, 32, 28, 28] for n = 8
def divide_batch(x, n):
    shape = get_tensor_shape(x)
    batch_size = shape[0]
    if isinstance(shape, list):
        new_shape = [n, batch_size // n] + shape[1:]
    else:
        new_batch_shape = tf.convert_to_tensor([n, batch_size // n])
        new_shape = tf.concat([new_batch_shape, shape[1:]], 0)
    result = tf.reshape(x, new_shape)
    return result


# MAIN IDEA
# calculate gradient for each sub_batch and average them for update
def divide_batch_train(inputs, labels, division_number, model, training, optimizer=None):
    if training and optimizer is None:
        raise ValueError("in training optimizer should be passed to training")
    inputs = tf.unstack(divide_batch(inputs, division_number), axis=0)
    labels = tf.unstack(divide_batch(labels, division_number), axis=0)
    model_weights = model.trainable_weights
    if training:
        gradients = [tf.zeros_like(w, dtype=w.dtype) for w in model_weights]
    loss = 0.0
    accuracy = 0.0
    operations = []
    for x, label in zip(inputs, labels):
        with tf.control_dependencies(operations):  # start new sub_batch after finishing last sub_batch
            probs = model(x, training=training)
            new_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label, probs))
            new_accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(label, probs))
            loss = loss + new_loss
            accuracy = accuracy + new_accuracy
            operations = [loss, accuracy]
            if training:
                new_grads = tf.gradients(loss, model_weights)
                gradients = [grad + g for grad, g in zip(gradients, new_grads)]
                operations = operations + gradients
    loss = loss / division_number
    accuracy = accuracy / division_number
    returns = [loss, accuracy]
    if training:
        gradients = [g / division_number for g in gradients]
        train = optimizer.apply_gradients(zip(gradients, model_weights))
        returns.append(train)
    return returns

data = np.load("../data/mnist/mnist.npy")
batch_size = 256 * 8
train_data = tf.data.Dataset.from_tensor_slices((data["train_images"], data["train_labels"]))
validation_data = tf.data.Dataset.from_tensor_slices((data["test_images"], data["test_labels"]))
train_data = train_data.shuffle(60000).repeat().batch(batch_size)
validation_data = validation_data.shuffle(10000).repeat().batch(batch_size)
train_steps = int(60000 / batch_size + 0.5)  # steps for one epoch
validation_steps = int(10000 / batch_size + 0.5)  # steps for one epoch
data = None

training_iterator = train_data.make_initializable_iterator()
validation_iterator = validation_data.make_initializable_iterator()
model = Model(name="model")
shapes = training_iterator.output_shapes
inputs = tf.placeholder(dtype=tf.float32, shape=shapes[0])
probs = model(inputs)  # only for buliding model. nothing more
optimizer = tf.train.AdamOptimizer(10e-5)

inputs, labels = training_iterator.get_next()
inputs = tf.cast(inputs, tf.float32)
labels = tf.cast(labels, tf.int32)
labels = tf.one_hot(labels, 10)
loss, accuracy, train = divide_batch_train(inputs, labels,
                                           division_number=8, model=model,
                                           training=True, optimizer=optimizer)
train_stuff = {"loss": loss, "accuracy": accuracy, "train": train}  # nodes for run

inputs, labels = validation_iterator.get_next()
inputs = tf.cast(inputs, tf.float32)
labels = tf.cast(labels, tf.int32)
labels = tf.one_hot(labels, 10)
loss, accuracy = divide_batch_train(inputs, labels, division_number=4, model=model, training=False)
validation_stuff = {"loss": loss, "accuracy": accuracy}  # nodes for run


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)

epochs = 400
sess.run([training_iterator.initializer, validation_iterator.initializer])
start_time = time.time()
for epoch in range(epochs):
    train_loss, train_size, train_accuracy = 0, 0, 0
    for _ in range(train_steps):
        new_size = batch_size
        new_loss, new_accuracy, _ = sess.run([train_stuff['loss'], train_stuff["accuracy"], train])
        train_loss = update_loss(train_loss, new_loss, train_size, new_size)
        train_accuracy = update_loss(train_accuracy, new_accuracy, train_size, new_size)
        train_size = train_size + new_size
    validation_loss, validation_accuracy, validation_size = 0, 0, 0
    for _ in range(validation_steps):
        new_size = batch_size
        new_loss, new_accuracy = sess.run([validation_stuff["loss"], validation_stuff["accuracy"]])
        validation_loss = update_loss(validation_loss, new_loss, validation_size, new_size)
        validation_accuracy = update_loss(validation_accuracy, new_accuracy, validation_size, new_size)
        validation_size = validation_size + new_size
    print("epoch: " + str(epoch + 1) + "\n")
    print("train loss: " + str(train_loss))
    print("train accuracy: " + str(train_accuracy))
    print("\n")
    print("validation loss: " + str(validation_loss))
    print("validation accuracy: " + str(validation_accuracy))
    print("\n\n")
end_time = time.time()
print((end_time - start_time) / 60)
