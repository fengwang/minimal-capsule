from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Input, Conv2D, Dense, Reshape, Layer, Lambda
from keras.models import Model
from keras.utils import to_categorical
from keras import initializers
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.datasets import mnist
import numpy as np
import tensorflow as tf

#
# The length of the output vector of a capsule is to represent the probability that the entity represented by the capsule
# is present in the current unit. A nonlinear squashing function ensures that
# - short vectors get shrunk to almost zero length and
# - long vectors get shrunk to a length slightly below 1
# this function is designed as
# v_j = \frac{||s_j||^2}{1 + ||s_j||^2 } \frac{s_j}{||s_j||}
#
def squash(output_vector, axis=-1):
    norm = tf.reduce_sum(tf.square(output_vector), axis, keep_dims=True)
    return output_vector * norm / ((1 + norm) * tf.sqrt(norm + 1.0e-10))

#
# This layer takes to input vectors:
#   - the first one is the output of the CapsuleLayer, 'n_calss' arrays
#   - the ground truth vector, an array with a length of 'n_class', with one of the elements is '1', the rests are '0'
#
class MaskingLayer(Layer):
    def call(self, inputs, **kwargs):
        input, mask = inputs
        return K.batch_dot(input, mask, 1)

    def compute_output_shape(self, input_shape):
        *_, output_shape = input_shape[0]
        return (None, output_shape)


#
# construct a conv layer, then reshape and apply squash operation
#
def PrimaryCapsule(n_vector, n_channel, n_kernel_size, n_stride, padding='valid'):
    def builder(inputs):
        output = Conv2D(filters=n_vector * n_channel, kernel_size=n_kernel_size, strides=n_stride, padding=padding)(inputs)
        output = Reshape( target_shape=[-1, n_vector], name='primary_capsule_reshape')(output)
        return Lambda(squash, name='primary_capsule_squash')(output)
    return builder

#
# Traditional Neural Network          Capsule
# scalar in scalar out       -->>     vector in vector out/matrix in matrix out
# back propagation update    -->>     routing update
#
class CapsuleLayer(Layer):
    def __init__(self, n_capsule, n_vec, n_routing, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.n_capsule = n_capsule
        self.n_vector = n_vec
        self.n_routing = n_routing
        self.kernel_initializer = initializers.get('he_normal')
        self.bias_initializer = initializers.get('zeros')

    def build(self, input_shape): # input_shape is a 4D tensor
        _, self.input_n_capsule, self.input_n_vector, *_ = input_shape
        self.W = self.add_weight(shape=[self.input_n_capsule, self.n_capsule, self.input_n_vector, self.n_vector], initializer=self.kernel_initializer, name='W')
        self.bias = self.add_weight(shape=[1, self.input_n_capsule, self.n_capsule, 1, 1], initializer=self.bias_initializer, name='bias', trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 2)
        input_tiled = tf.tile(input_expand, [1, 1, self.n_capsule, 1, 1])
        input_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]), elems=input_tiled, initializer=K.zeros( [self.input_n_capsule, self.n_capsule, 1, self.n_vector]))
        for i in range(self.n_routing): # routing
            c = tf.nn.softmax(self.bias, dim=2)
            outputs = squash(tf.reduce_sum( c * input_hat, axis=1, keep_dims=True))
            if i != self.n_routing - 1:
                self.bias += tf.reduce_sum(input_hat * outputs, axis=-1, keep_dims=True)
        return tf.reshape(outputs, [-1, self.n_capsule, self.n_vector])

    def compute_output_shape(self, input_shape):
        # output current layer capsules
        return (None, self.n_capsule, self.n_vector)

#
# This layer takes 'n_class' arrays as input, outputs an array of size 'n_class',
# each eleemnt in the output array represent the possibility,
# i.e., the last layer in Figure 2.
#
class LengthLayer(Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1, keep_dims=False))

    def compute_output_shape(self, input_shape):
        *output_shape, _ = input_shape
        return tuple(output_shape)


#
# margin loss is employed to measure the accuracy of the capsule net,
# in the code below, mean absolute error is used to measure the accuracy of the reconstructed image
#
def margin_loss(y_ground_truth, y_prediction):
    _m_plus = 0.9
    _m_minus = 0.1
    _lambda = 0.5
    L = y_ground_truth * tf.square(tf.maximum(0., _m_plus - y_prediction)) + _lambda * ( 1 - y_ground_truth) * tf.square(tf.maximum(0., y_prediction - _m_minus))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


if __name__ == "__main__":

    # preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)

    # make model
    input_shape = [28, 28, 1]
    n_class = 10
    n_routing = 3

    # encoder
    x = Input(shape=input_shape)
    conv1 = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primary_capsule = PrimaryCapsule( n_vector=8, n_channel=32, n_kernel_size=9, n_stride=2)(conv1)
    digit_capsule = CapsuleLayer( n_capsule=n_class, n_vec=16, n_routing=n_routing, name='digit_capsule')(primary_capsule)
    output_capsule = LengthLayer(name='output_capsule')(digit_capsule)

    # decoder
    mask_input = Input(shape=(n_class, ))
    mask = MaskingLayer()([digit_capsule, mask_input])  # two inputs
    dec = Dense(512, activation='relu')(mask)
    dec = Dense(1024, activation='relu')(dec)
    dec = Dense(784, activation='sigmoid')(dec)
    dec = Reshape(input_shape)(dec)

    model = Model([x, mask_input], [output_capsule, dec])
    plot_model(model, 'capsule.png', show_shapes=True, rankdir='TB')
    model.summary()
    model.compile(optimizer='adam', loss=[ margin_loss, 'mae' ], metrics=[ margin_loss, 'mae'])

    # train capsule model
    model.fit([X, Y], [Y, X], batch_size=128, epochs=50, validation_split=0.2)
    model.save_weights('capsule_trained.h5')

