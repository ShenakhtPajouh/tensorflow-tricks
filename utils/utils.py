import tensorflow as tf

def get_tensor_shape(x):
    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)
    if static_shape is None:
        return dynamic_shape
    shape = []
    for i, sp in enumerate(static_shape):
        if sp is None:
            shape.append(dynamic_shape[i])
        else:
            shape.append(sp)
    return shape


def update_loss(prev_loss, new_loss, prev_size, new_size):
    coef = prev_size / (prev_size + new_size)
    loss = coef * prev_loss + (1 - coef) * new_loss
    return loss




