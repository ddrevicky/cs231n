from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)   # DxH
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes) # HxC
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        params = self.params
        W1, W2, b1, b2 = params['W1'], params['W2'], params['b1'], params['b2'] 
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        Z1, cache_Z1 = affine_forward(X, W1, b1)
        A1, cache_A1 = relu_forward(Z1)
        scores, cache_scores = affine_forward(A1, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
        
        dA1, dW2, db2 = affine_backward(dscores, cache_scores)
        dW2 += self.reg * W2
        
        dZ1 = relu_backward(dA1, cache_A1)
        
        _, dW1, db1 = affine_backward(dZ1, cache_Z1)
        dW1 += self.reg * W1
        
        grads.update({ 'W1':dW1, 'W2':dW2, 'b1':db1, 'b2':db2 })
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def batchnorm_relu_forward(x, gamma, beta, bn_param):
    """
    Convenience layer that performs batch normalization transform followed by a ReLU

    Input:
    - x: Input to the batch norm layer
    - gamma, beta, bn_param: Parameters for the batch norm layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, bc_cache = batchnorm_forward(x, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a)
    cache = (bc_cache, relu_cache)
    return out, cache
    
def batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the batchnorm ReLU convenience layer.
    """
    bc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dgamma, dbeta = batchnorm_backward(da, bc_cache)
    return dx, dgamma, dbeta
    
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        sizes = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, len(sizes)):
            self.params['W' + str(i)] = weight_scale * np.random.randn(sizes[i-1], sizes[i])
            self.params['b' + str(i)] = np.zeros(sizes[i])
            if use_batchnorm and i < len(sizes) - 1: # Batch norm is not applied to the last layer
                self.params['gamma' + str(i)] = np.ones(sizes[i])
                self.params['beta' + str(i)] = np.zeros(sizes[i])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        # Convenience variables
        N = X.shape[0]
        params = self.params
        num_layers = self.num_layers
        reg = self.reg
        # Note that the parameter numbering starts at 1 (not 0), therefore we also index from 1
        Ws, bs, gammas, betas = [None], [None], [None], [None]
        for i in range(1, num_layers + 1): 
            Ws.append(params['W' + str(i)])
            bs.append(params['b' + str(i)])
            if self.use_batchnorm and i < num_layers: # Batch norm is not applied to the last layer
                gammas.append(params['gamma' + str(i)])
                betas.append(params['beta' + str(i)])
        
        # Forward propagation
        As, Zs, Z_caches, A_caches = [X.reshape(N, -1)], [None], [None], [None]
        As_drop, A_drop_caches = [None], [None]
        
        for i in range(1, num_layers + 1):
            A = As[i - 1]
            if self.use_dropout and i > 1:
                A = As_drop[i - 1]
            Z, Z_cache = affine_forward(A, Ws[i], bs[i])
            Zs.append(Z)
            Z_caches.append(Z_cache)
            if i < num_layers:
                if self.use_batchnorm:
                    A, A_cache = batchnorm_relu_forward(Z, gammas[i], betas[i], self.bn_params[i - 1])
                else:
                    A, A_cache = relu_forward(Z)
                As.append(A)
                A_caches.append(A_cache)
                if self.use_dropout:
                    A_drop, A_drop_cache = dropout_forward(A, self.dropout_param)
                    As_drop.append(A_drop)
                    A_drop_caches.append(A_drop_cache)
        
        scores = Zs[-1]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        # Loss
        loss, dscores = softmax_loss(scores, y)
        for i in range(1, num_layers + 1):
            loss += 0.5 * reg * (Ws[i]**2).sum()
        
        # Backpropagation
        dA = dscores
        for i in range(num_layers, 0, -1):
            dZ = dA
            if i < num_layers:
                if self.use_dropout:
                    dA = dropout_backward(dA, A_drop_caches[i])
                if self.use_batchnorm:
                    dZ, dgamma, dbeta = batchnorm_relu_backward(dA, A_caches[i])
                    grads['gamma' + str(i)] = dgamma
                    grads['beta' + str(i)] = dbeta
                else:
                    dZ = relu_backward(dA, A_caches[i])
                
            dA, dW, db = affine_backward(dZ, Z_caches[i])
            dW += reg * Ws[i]
            
            grads['W' + str(i)] = dW
            grads['b' + str(i)] = db
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
