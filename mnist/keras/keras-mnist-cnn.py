from __future__ import print_function

import argparse
import datetime

import keras
import tensorflow as tf
from keras import backend as K
from keras.optimizers import RMSprop

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)


def train(data_dir, logdir, output_model_dir, batch_size=128, epochs=10, num_classes=10):
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(data_dir, one_hot=False)

    x_train = mnist.train.images.reshape(55000, 784)
    x_test = mnist.test.images.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(mnist.train.labels, num_classes)
    y_test = keras.utils.to_categorical(mnist.test.labels, num_classes)

    print(x_train.shape, y_train.shape)

    # model
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # tensorboard callback
    logdir = logdir + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, batch_size=batch_size)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(output_model_dir)


if __name__ == '__main__':
    num_classes = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='/output/logs',
                        help='tensorboard output log dir.')
    parser.add_argument('--data_dir', type=str, default='/data',
                        help='dataset dir.')
    parser.add_argument('--output_model_dir', type=str, default='/output/models/mymodel.h5',
                        help='model dir.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs.')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='num of classes.')
    args, unparsed = parser.parse_known_args()
    train(args.data_dir, args.logdir, args.output_model_dir, args.batch_size, args.epochs, args.num_classes)
