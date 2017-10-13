from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        layers = 3
        self.params["W1"] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params["b1"] = np.zeros(num_filters)
        W2_input_dim = num_filters * H / 2 * W /2
        self.params["W2"] = np.random.randn(W2_input_dim, hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)
        self.params["W3"] = np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)
            
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        #print(X.shape)
        out1, cache1 = conv_forward_fast(X, W1, b1, conv_param)
        #print(out.shape)
        out2, cache2 = relu_forward(out1)
        out3, cache3 = max_pool_forward_fast(out2, pool_param)
        #print(out.shape)
        out4 = np.reshape(out3, (X.shape[0], -1))
        #print(out.shape, W2.shape, b2.shape)
        out5, cache4 = affine_forward(out4, W2, b2)
        out6, cache5 = relu_forward(out5)
        scores, cache6 = affine_forward(out6, W3, b3)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, scores_prob = softmax_loss(scores, y)
        dx, grads["W3"], grads["b3"] = affine_backward(scores_prob, cache6)
        dx = relu_backward(dx, cache5)
        dx, grads["W2"], grads["b2"] = affine_backward(dx, cache4)
        dx = dx.reshape(out3.shape)
        dx = max_pool_backward_fast(dx, cache3)
        dx = relu_backward(dx, cache2)
        dx, grads["W1"], grads["b1"] = conv_backward_fast(dx, cache1)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
