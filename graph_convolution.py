from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer
from keras.layers.core import *

import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('tf')


class GraphConv(Layer):
    def __init__(self,
                 filters,
                 num_neighbors,
                 neighbors_ix_mat,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        # 检测后台是不是基于tensorflow
        if K.backend() != 'tensorflow':
            raise Exception("GraphConv with Tensorflow Backend.")
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConv, self).__init__(**kwargs)
        # 传回参数
        self.filters = filters
        self.num_neighbors = num_neighbors
        self.neighbors_ix_mat = neighbors_ix_mat

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        input_dim = input_shape[2]
        kernel_shape = (self.num_neighbors, input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, x):
        x_expanded = tf.gather(x, self.neighbors_ix_mat, axis=1)
        # x_expanded = x[:,self.neighbors_ix_mat,:]
        output = tf.tensordot(x_expanded, self.kernel, [[2, 3], [0, 1]])
        if self.use_bias:
            output += tf.reshape(self.bias, (1, 1, self.filters))

        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)

