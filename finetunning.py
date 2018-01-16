
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression


def vgg19(num_classes):
    
    
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


    fc1 = fully_connected(flatten_layer, 4096, activation='relu')
    dp1 = dropout(fc1, 0.5)
    # layer below this are not restored!
    fc2 = fully_connected(dp1, 4096, activation='relu', restore= False)
    dp2 = dropout(fc2, 0.5)

    network = fully_connected(dp2, num_classes, activation='softmax', restore=False)

    regression = tflearn.regression(network, optimizer='adam',
                                loss='categorical_crossentropy',
                                learning_rate=0.001, restore=False)
    
    model = tflearn.DNN(regression, checkpoint_path='vgg-finetuning',
                        tensorboard_dir="./logs")

    model.load("vgg19.tflearn", weights_only=True)
    
    return model
    

