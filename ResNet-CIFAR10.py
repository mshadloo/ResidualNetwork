import numpy as np
import sys
import os
import tensorflow as tf
import keras as K
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model

from keras.initializers import glorot_uniform
from keras.datasets import cifar10
from keras import optimizers,utils
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR10.")
    parser.add_argument('--model', nargs='?', default='',
                        help='pretrained model')
    parser.add_argument('--res_block_num', type = int, default=5,
                        help='number of residual block of each stack')
    parser.add_argument('--epochs', type=int, default=82,
                        help='Number of epochs.')

    return parser.parse_args()

def resnet(input_shape = (32, 32, 3), classes = 10, n = 1):
    def residual_block(X, filters_num, f = 3, convolution_block=False):
        # channel_num
        X_shortcut = X
        if not convolution_block:

            if X.shape[3] != filters_num:
                raise ValueError("for identical block number of filters and number of input channel must be the same")
            stride = 1
        else:
            stride = 2
        X = Conv2D(filters=filters_num, kernel_size=(f, f), strides=(stride, stride), padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filters_num, kernel_size=(f, f), strides=(1, 1), padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        if convolution_block:
            X_shortcut = Conv2D(filters=filters_num, kernel_size=(1, 1), strides=(stride, stride), padding='same',
                                kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
            X_shortcut = BatchNormalization(axis=3)(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        return X

    X_input = Input(input_shape)
    X = Conv2D(16, (3, 3), strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=0),padding ="same")(X_input)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    # first stack of residual blocks, output is 32 x 32 x 16
    for i in range(n):
        X = residual_block(X, 16)

    # second stack of residual blocks, output is 16 x 16 x 32
    X = residual_block(X, 32, convolution_block= True)
    for i in range(1, n):
        X = residual_block(X, 32)

    # third stack of residual blocks, output is 64 x 8 x 8
    X = residual_block(X,64, convolution_block= True)
    for i in range(1, n):
        X = residual_block(X, 64)

    # average pooling
    X = GlobalAveragePooling2D()(X)

    # fully connected layer
    # X = Flatten()(X)
    X_output = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input, outputs=X_output, name='ResNet'+str(6*n+2))

    return model




def main(res_block_num= 5, num_epochs = 82, pretrained_model = None):
    num_classes = 10
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # Convert training and test labels to one hot matrices
    y_train = utils.to_categorical(y_train, )
    y_test = utils.to_categorical(y_test, num_classes)
    loss_train,acc_train, loss_test, acc_test = [], [] , [] ,[]

    if not pretrained_model:
        lr = 0.1
        print("Building model...")
        model = resnet(input_shape = (32, 32, 3), classes = 10)
        optimizer = optimizers.Adam(lr = lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            st = time.time()
            # shuffle training data
            train_indices = np.arange(x_train.shape[0])
            np.random.shuffle(train_indices)
            x_train = x_train[train_indices, :, :, :]
            y_train = y_train[train_indices]

            # In each epoch, we pass over all training data:

            history = model.fit(x_train, y_train,
                      batch_size=128,
                      epochs=1,
                      shuffle=True, verbose=0)
            print("epoch: ", epoch)
            loss_train.append(history.history["loss"])
            acc_train.append(history.history["acc"])
            print("Train Loss = " + str(history.history["loss"]))
            print("Train Accuracy = " + str(history.history["acc"]))
            preds = model.evaluate(x_test, y_test,batch_size=500, verbose=0)
            loss_test.append(preds[0])
            acc_test.append(preds[1])
            print("Loss = " + str(preds[0]))
            print("Test Accuracy = " + str(preds[1]))
            print(time.time()-st)
            if (epoch + 1) == 41 or (epoch + 1) == 61:
                lr = lr * 0.1
                K.set_value(model.optimizer.learning_rate, lr)
                print("New LR:" + str(lr))

            model.save("resnet" + str(6 * res_block_num + 2) + "_cifar10.h5")
        print("lost_train:"+ loss_train)
        print("acc_train:" + acc_train)
        print("lost_test:" + loss_test)
        print("acc_test:" + loss_test)
    else:
        model = load_model(pretrained_model)
    preds = model.evaluate(x_test, y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on cifar-10 using Keras.")
        print("Network architecture and training parameters are as in section 4.2 in 'Deep Residual Learning for Image Recognition'.")
        print("Usage: %s [N [MODEL]]" % sys.argv[0])
        print()
        print("N: Number of stacked residual building blocks per feature map (default: 5)")
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        args = parse_args()
        kwargs = {}
        kwargs['res_block_num'] = args.res_block_num
        kwargs['pretrained_model'] = args.model
        kwargs['num_epochs'] = args.epochs

        main(**kwargs)
