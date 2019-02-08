#!/usr/bin/env python3.6
import ssl
from time import time

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import to_categorical

from util import *
import ssl
from tensorflow.keras.optimizers import SGD

ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.applications.vgg19 import VGG19
from Logging import *


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


if __name__ == '__main__':
    # Define upper level settings
    hyper_parameters = {'epochs': 600, 'name': 'Custom', 'n_classes': 3, 'use_weights':True, 'sanity_test':True}
    model_dir = f'./run_logs/{time()}_{"_".join("{!s}.{!r}".format(key,val) for (key,val) in hyper_parameters.items())}'

    STANDARD_IMAGE_SIZE = (224, 224, 3)

    # Get the directory path
    data_dir = get_absolute_data_path()

    X = None
    Y = []
    target_dict = {name: i for i, name in enumerate(os.listdir(data_dir))}
    print(f'Loading Images')
    # For each class
    for directory in os.listdir(data_dir):
        # For each image in that class
        for filename in os.listdir(data_dir + directory):
            # Get a standard sized image
            im = np.array(Image.open(data_dir + directory + os.sep + filename)
                          .resize(STANDARD_IMAGE_SIZE[0:2], Image.ANTIALIAS))
            # im = rgb2gray(im)
            # plt.imshow(im)
            # plt.show()
            # Get the actual class for that image
            Y.append(target_dict[directory])

            # Build the X sample array via stacking
            if X is not None:
                X = np.vstack((X, im[np.newaxis, ...]))
                # break
            else:
                X = np.array([im])

    # Fix X
    # Scale it, and flatten
    # X = np.array([np.array(image / 255).flatten() for image in X])
    # Fix Y
    Y = np.array(Y).reshape(-1, 1)
    Y = to_categorical(Y, num_classes=hyper_parameters['n_classes'])

    if hyper_parameters['sanity_test']:
        X_train, X_test, y_train, y_test = (X, X, Y, Y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    print("Done building data set")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=STANDARD_IMAGE_SIZE),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(hyper_parameters['n_classes'], activation=tf.nn.softmax)
    ])
    # model = VGG19(include_top=False, weights=None,
    #               input_tensor=None, input_shape=(224, 224, 3), pooling=None)
    #
    # # If we want to use weights, then try to load them
    # if hyper_parameters['use_weights']:
    #     model.load_weights('./weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    #
    # """ Freeze the previous layers """
    # for layer in model.layers:
    #     layer.trainable = False
    #
    # """ By Setting top to False, we need to add our own classification layers """
    # # The model documentation notes that this is the size of the classification block
    # x = Flatten(name='flatten')(model.output)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dense(hyper_parameters['n_classes'], activation='softmax', name='predictions')(x)
    #
    # # create graph of your new model
    # model = Model(inputs=model.inputs, outputs=x, name='vgg19')

    opt = SGD(lr=0.01)
    model.compile(optimizer=opt,  # 'adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'mean_squared_error'])

    tensorboard = TrainValTensorBoard(log_dir=model_dir, histogram_freq=1, X_train=X_train,
                                      X_test=X_test, y_train=y_train, y_test=y_test,
                                      write_graph=True, write_images=True)

    model.fit(X_train, y_train, epochs=hyper_parameters['epochs'],
              validation_data=(X_test, y_test),
              callbacks=[tensorboard])
    print(f'\nEvaluation: {model.evaluate(X_test, y_test)}')  # So this is currently: loss & accuracy
    print(f'\nFor Y targets {y_test}')

    # Save entire model to a HDF5 file
    model.save(model_dir + '/model.h5')

    # # Setup the session
    # sess = tf.InteractiveSession()
    #
    # tf_X = tf.placeholder(tf.float32, [None, 200*200], name='tf_X')
    # tf_Y = tf.placeholder(tf.int64, shape=[None, ], name='tf_Y')
    # print("Build network")
    # network = tl.layers.InputLayer(tf_X, name='input')
    # network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    # network = tl.layers.DenseLayer(network, n_units=200*200, act=tf.nn.relu, name='relu1')
    # network = tl.layers.DropoutLayer(network, keep=0.8, name='drop2')
    # network = tl.layers.DenseLayer(network, n_units=3, act=tf.nn.relu, name='relu2')
    #
    # print("Set up the metric ")
    # y = network.outputs
    # cost = tl.cost.cross_entropy(y, tf_Y, 'cost')
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf_Y)
    # acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # y_op = tf.argmax(tf.nn.softmax(y), 1)
    #
    # print("Define the optimizer ")
    # train_params = network.all_params
    # train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
    #                                   epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
    #
    # print("Initialize all variables in the session ")
    # tl.layers.utils.initialize_global_variables(sess)
    #
    # print("Print network information")
    # network.print_params()
    # network.print_layers()
    #
    # print("Training Network")
    # tl.utils.fit(sess, network, train_op, cost, X_train, y_train, tf_X, tf_Y,
    #              acc=acc, batch_size=500, n_epoch=500, print_freq=5,
    #              X_val=X_train, y_val=y_train, eval_train=False)
    #
    # # evaluation
    # tl.utils.test(sess, network, acc, X_test, y_test, tf_X, tf_Y, batch_size=None, cost=cost)
    #
    # # save the network to .npz file
    # tl.files.save_npz(network.all_params, name='model.npz')
    # sess.close()
