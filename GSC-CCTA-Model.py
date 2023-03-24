# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from keras import backend as K, initializers
from keras.engine import Layer, InputSpec

K.set_image_dim_ordering('tf')

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Dropout
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization, Permute, Conv2D, AveragePooling2D, concatenate, activations, Bidirectional, GRU, Flatten, LSTM, add, Multiply, Activation
from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, Merge, concatenate, activations, merge, add, Multiply
from keras.callbacks import TensorBoard
from keras.initializers import Initializer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from data import image_size_dict

from spatialattention import SpatialAttention
from crossattention import CrossAttention
from secondpooling import SecondOrderPooling
from tensorflow.keras import layers

from graph_convolution import GraphConv
from GSC_utils import generate_Q


def get_model(img_rows, img_cols, num_PC, nb_classes, dataID=1, type='GSC_CCTA', lr=0.01):
    if num_PC == 0:
        num_PC = image_size_dict[str(dataID)][2]
    if type == 'GSC_CCTA':
        model = GSC_CCTA(img_rows, img_cols, num_PC, nb_classes)
    else:
        print('invalid model type, default use GSC_CCTA model')
        model = GSC_CCTA(img_rows, img_cols, num_PC, nb_classes)

    rmsp = RMSprop(lr=lr, rho=0.9, epsilon=1e-05)
    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return model

class Symmetry(Initializer):
    """N*N*C Symmetry initial
    """
    def __init__(self, n=200, c=16, seed=0):
        self.n = n
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rv = K.truncated_normal([self.n, self.n, self.c], 0., 1e-5, dtype=dtype, seed=self.seed)
        rv = (rv + K.permute_dimensions(rv, pattern=(1, 0, 2))) / 2.0
        return K.reshape(rv, [self.n * self.n, self.c])



def stdpooling(x):
    std = K.std(x, axis=1, keepdims=False)
    return std



class KnowledgeLayer(Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activations.get(activation)
        super(KnowledgeLayer, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)
        X, A1,A2 = x
        A = A1 + A2 + X

        # concatenate2 = K.concatenate([A, X], axis=3)
        return A
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0],image_size,image_size, 1 * self.units )
        return input_dim


class ACD(Layer):

    def __init__(self, units,Thr, activation=None, **kwargs):
        self.units = units
        self.Thr = Thr
        self.activation = activations.get(activation)
        super(ACD, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)
        H1,V1 ,H2,V2 ,X1,X2= x

        cos1 = tf.multiply(tf.sqrt(tf.multiply(H1, H1)), tf.sqrt(tf.multiply(V1, V1)))
        cosin1 = tf.multiply(H1, V1) / cos1

        cos2 = tf.multiply(tf.sqrt(tf.multiply(H2, H2)), tf.sqrt(tf.multiply(V2, V2)))
        cosin2 = tf.multiply(H2, V2) / cos2

        Zeos = tf.zeros_like(X1)
        Ones = tf.ones_like(X1)

        print(self.Thr)

        Y = tf.where(cosin1 > self.Thr, x=Ones, y=Zeos)

        Y1 = tf.where(cosin2 > self.Thr, x=Ones, y=Zeos)

        concatenate = K.concatenate([Y * X1,  Y1 * X2], axis=3)

        return [concatenate]
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0],image_size,image_size, 2 * self.units )
        return [input_dim,input_dim]



num_neighbors1 = 24
num_neighbors2 = 24
num_neighbors3 = 6
num_neighbors4 = 3


q_mat_layer1 = generate_Q(8,4)
q_mat_layer1 = np.argsort(q_mat_layer1,1)[:,-num_neighbors1:]

q_mat_layer2 = generate_Q(6,4)
q_mat_layer2 = np.argsort(q_mat_layer2,1)[:,-num_neighbors2:]

q_mat_layer3 = generate_Q(4,4)
q_mat_layer3 = np.argsort(q_mat_layer3,1)[:,-num_neighbors3:]

q_mat_layer4 = generate_Q(2,4)
q_mat_layer4 = np.argsort(q_mat_layer4,1)[:,-num_neighbors4:]



def attention_vertical(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1,2))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)


    a_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return a_probs


def attention_horizontal(inputs2):
    input_dim1 = int(inputs2.shape[1])
    input_dim2 = int(inputs2.shape[2])
    input_dim3 = int(inputs2.shape[3])

    a = Permute((3, 2,1))(inputs2)
    a = Reshape((input_dim3, input_dim2,input_dim1 ))(a) # this line is not useful. It's just to know which dimension is what.

    a = Dense(input_dim2, activation='softmax')(a)

    b_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return b_probs


Thr=0.7
def GSC_CCTA(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    #
    F = Reshape([img_rows * img_cols, num_PC])(CNNInput)
    F = Activation('relu')(F)
    F = BatchNormalization()(F)

    # #L2 Norm
    # F = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='f2')(F)

    #
    F2 = SecondOrderPooling(name='feature1')(F)
    F2 = Reshape((270, 270))(F2)  #按光谱长修改


    #
    gcn1 = GraphConv(filters=64, neighbors_ix_mat=q_mat_layer4, num_neighbors=3, activation='elu')(F)
    gcn2 = GraphConv(filters=64, neighbors_ix_mat=q_mat_layer4, num_neighbors=3, activation='elu')(F2)

    G = concatenate([gcn1, gcn2], axis=1)


    T1 = Bidirectional(GRU(128, return_sequences=False))(gcn1)
    T2 = Bidirectional(GRU(128, return_sequences=False))(gcn2)

    T1 = Reshape((2, 2, 64))(T1)
    T2 = Reshape((2, 2, 64))(T2)

    #
    #
    h1 = attention_horizontal(T1)
    h2 = attention_horizontal(T2)

    #
    V1 = attention_vertical(T1)
    V2 = attention_vertical(T2)
    #
    #
    k1 = KnowledgeLayer(64)([T1, h1, V1])
    k2 = KnowledgeLayer(64)([T2, h2, V2])

    k = ACD(64,Thr)([h1, V1, h2, V2, k1, k2])

    # k = Activation('relu')(k)
    k = Reshape((2, 2, 128))(k)
    # k = BatchNormalization()(k)
    # k = concatenate([k1, k2], axis=1)


    k = Reshape((8,64))(k)

    #
    F = concatenate([G, k], axis=1)

    F = Lambda(stdpooling)(F)

    print(F)

    n = math.ceil(math.sqrt(K.int_shape(F)[-1]))
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=n, c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=[F])

    return model