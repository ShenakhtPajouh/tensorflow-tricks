# Train a Model with while_loop

This is a weird trick to do some steps of model training in a while_loop instead of python for.
The main idea is to replace python for loop of training with a while_loop and construct the whole training process in
the GraphDef of tensorflow, hence it can speed up your training process if the pythonic operations between each two steps
of the training is expensive.

Practically it is not a good idea to implement the whole training process in Tensorflow Graph because we want to save the checkpoints
after some steps and also tf.print does not work in jupyter notebook, since jupyter only shows python session outputs. However
we implement both partial and full version of implementing training part in tensorflow graph.

This will also give an insight to what will be going on in Tensorflow 2. 

  
