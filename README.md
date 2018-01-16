# tflearn VGG19


In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest. 

The three major Transfer Learning scenarios look as follows:

- **ConvNet as fixed feature extractor**. Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. In an AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features **CNN codes**. It is important for performance that these codes are ReLUd (i.e. thresholded at zero) if they were also thresholded during the training of the ConvNet on ImageNet (as is usually the case). Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.
- **Fine-tuning the ConvNet**. The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In case of ImageNet for example, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.
- **Pretrained models**. Since modern ConvNets take 2-3 weeks to train across multiple GPUs on ImageNet, it is common to see people release their final ConvNet checkpoints for the benefit of others who can use the networks for fine-tuning. For example, the Caffe library has a [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) where people share their network weights. [1](#1)



Unfortunately some libraries doesn't have pretrained models like tflearn. In this repository I wanted to share And here is the pretrained VGG19 network ready to finetune.

Weights can be downloaded from the link below:

[Weights](https://www.dropbox.com/s/3rjode6oqhqtq0e/Archive.zip?dl=0)



# Recommended use

VGG19 net has a lot of parameters and take a lot of memory so it is better to run through network until the point you want to fine tune. Than save it, and load from memory directly for next iterations. 

```python

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Building 'VGG Network'
input_layer = input_data(shape=[None, 224, 224, 3])

block1_conv1 = conv_2d(input_layer, 64, 3, activation='relu', name='block1_conv1')
block1_conv2 = conv_2d(block1_conv1, 64, 3, activation='relu', name='block1_conv2')
block1_pool = max_pool_2d(block1_conv2, 2, strides=2, name = 'block1_pool')

block2_conv1 = conv_2d(block1_pool, 128, 3, activation='relu', name='block2_conv1')
block2_conv2 = conv_2d(block2_conv1, 128, 3, activation='relu', name='block2_conv2')
block2_pool = max_pool_2d(block2_conv2, 2, strides=2, name = 'block2_pool')

block3_conv1 = conv_2d(block2_pool, 256, 3, activation='relu', name='block3_conv1')
block3_conv2 = conv_2d(block3_conv1, 256, 3, activation='relu', name='block3_conv2')
block3_conv3 = conv_2d(block3_conv2, 256, 3, activation='relu', name='block3_conv3')
block3_conv4 = conv_2d(block3_conv3, 256, 3, activation='relu', name='block3_conv4')
block3_pool = max_pool_2d(block3_conv4, 2, strides=2, name = 'block3_pool')

block4_conv1 = conv_2d(block3_pool, 512, 3, activation='relu', name='block4_conv1')
block4_conv2 = conv_2d(block4_conv1, 512, 3, activation='relu', name='block4_conv2')
block4_conv3 = conv_2d(block4_conv2, 512, 3, activation='relu', name='block4_conv3')
block4_conv4 = conv_2d(block4_conv3, 512, 3, activation='relu', name='block4_conv4')
block4_pool = max_pool_2d(block4_conv4, 2, strides=2, name = 'block4_pool')

block5_conv1 = conv_2d(block4_pool, 512, 3, activation='relu', name='block5_conv1')
block5_conv2 = conv_2d(block5_conv1, 512, 3, activation='relu', name='block5_conv2')
block5_conv3 = conv_2d(block5_conv2, 512, 3, activation='relu', name='block5_conv3')
block5_conv4 = conv_2d(block5_conv3, 512, 3, activation='relu', name='block5_conv4')
block4_pool = max_pool_2d(block5_conv4, 2, strides=2, name = 'block4_pool')
flatten_layer = tflearn.layers.core.flatten (block4_pool, name='Flatten')
regression = tflearn.regression(flatten_layer, optimizer='adam',
                            loss='categorical_crossentropy',
                            learning_rate=0.001, restore=False)

model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning',
                    tensorboard_dir="./logs")
model.load("vgg19.tflearn", weights_only=True)

X_features = model.predict(X_train)


# New Building deep neural network
input_layer = tflearn.input_data(shape=[None, X_features.shape[1]])
dense1 = tflearn.fully_connected(input_layer, 512, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)
dense2 = tflearn.fully_connected(dropout1, 512, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)
softmax = tflearn.fully_connected(dropout2, num_classes, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")


```





## 1 http://cs231n.github.io/transfer-learning/ 

