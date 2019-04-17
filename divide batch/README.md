# Divide Batch into Sub-Batches

It is observed by experiments that most deep learning models can converge properly with mini-batch size between 32-512.
But in many cases, lack of memory resources prevent us from choosing such mini-batch sizes.
for instance it doesn't seem possible to train a 12 Layer Transofromer (like Bert on GPU with 11GB memory) with mini-batch size more than 8.
In this code we show that how we can divide a mini-batch into sub-batches and then avererage the gradients for these sub-batches in order to train model with arbitary mini-batch size.

__note__: for simplicity these model is a classifier for MNIST dataset. so first preprare dataset on ```../data/minist``` directory.
 
