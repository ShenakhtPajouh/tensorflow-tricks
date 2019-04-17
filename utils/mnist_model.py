import tensorflow as tf
from utils import get_tensor_shape

class Model(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), name="Conv_1")
        self.max_pool = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling")
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), name="Conv_2")
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', name="dense_1")
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax', name="dense_2")

    def call(self, inputs, training=None, dropout=0.1):
        if training is None:
            training = True
        training = tf.convert_to_tensor(training)
        x = tf.expand_dims(inputs, -1)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = tf.keras.activations.relu(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = tf.keras.activations.relu(x)
        shape = get_tensor_shape(x)
        new_shape = (shape[0], shape[1] * shape[2] * shape[3])
        x = tf.reshape(x, new_shape)
        x = tf.cond(training, lambda: tf.nn.dropout(x, dropout), lambda: x)
        x = self.dense1(x)
        x = tf.cond(training, lambda: tf.nn.dropout(x, dropout), lambda: x)
        x = self.dense2(x)
        return x

    def __call__(self, inputs, training=None, dropout=0.1):
        return super().__call__(inputs=inputs, training=training, dropout=dropout)
