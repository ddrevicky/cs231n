from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_flat = np.reshape(x, (x.shape[0], -1))
    out = np.dot(x_flat, w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)

    N = x.shape[0]
    dw = np.dot(x.reshape((N, -1)).T, dout)
    
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x >= 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        
        # Note: I have tried to use the same variables names as used by the authors in the original 
        # Batch Normalization paper (2015, Ioffe and Szegedy)
        x_sum = np.sum(x, axis=0)
        mean = x_sum / N
        
        x_min_mean = x - mean
        x_min_mean_sq = x_min_mean**2
        x_min_mean_sq_sum = np.sum(x_min_mean_sq, axis=0)
        variance = x_min_mean_sq_sum / N
        variance_pl_eps = variance + eps
        variance_pl_eps_sqrt = np.sqrt(variance_pl_eps)
        inv_variance_pl_eps_sqrt = 1.0 / variance_pl_eps_sqrt
        
        x_roof = x_min_mean * inv_variance_pl_eps_sqrt
        x_roof_gamma = gamma * x_roof
        y = x_roof_gamma + beta
        
        out = y
        cache = (x, gamma, x_sum, mean, x_min_mean, x_min_mean_sq, x_min_mean_sq_sum,
                 variance, variance_pl_eps, variance_pl_eps_sqrt, inv_variance_pl_eps_sqrt,
                 x_roof, x_roof_gamma)
        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * variance
        ######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_roof = (x - running_mean) / np.sqrt((running_var + eps))
        y = gamma * x_roof + beta
        out = y
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, gamma, x_sum, mean, x_min_mean, x_min_mean_sq, x_min_mean_sq_sum, \
    variance, variance_pl_eps, variance_pl_eps_sqrt, inv_variance_pl_eps_sqrt, \
    x_roof, x_roof_gamma = cache
    
    dx = np.zeros_like(x)
    m = x.shape[0]
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    dbeta = np.sum(dout * 1.0, axis=0)
    dgamma = np.sum(dout * x_roof, axis=0)
    
    dx_roof_gamma = dout * 1.0
    dx_roof = dx_roof_gamma * gamma
    dx_min_mean = dx_roof * inv_variance_pl_eps_sqrt
    dx += dx_min_mean
    
    dmean = np.sum(dx_min_mean * -1.0, axis=0)
    dx_sum = dmean * (1.0 / m)
    dx += dx_sum
    
    dinv_variance_pl_eps_sqrt = dx_roof * x_min_mean
    dvariance_pl_eps_sqrt = dinv_variance_pl_eps_sqrt * (-1.0 / variance_pl_eps_sqrt)
    dvariance_pl_eps = dvariance_pl_eps_sqrt * (1.0 / (2.0 * variance_pl_eps))
    dvariance = np.sum(dvariance_pl_eps * 1.0, axis=0)
    dx_min_mean_sq_sum = dvariance * (1.0 / m)
    dx_min_mean_sq = dx_min_mean_sq_sum * 1.0
    dx_min_mean = dx_min_mean_sq * (2.0 * x_min_mean)
    dx += dx_min_mean * 1.0     
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    x, gamma, x_sum, mean, x_min_mean, x_min_mean_sq, x_min_mean_sq_sum, \
    variance, variance_pl_eps, variance_pl_eps_sqrt, inv_variance_pl_eps_sqrt, \
    x_roof, x_roof_gamma = cache
    dx = np.zeros_like(x)
    m = x.shape[0]
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    # Note: These equations directly correspond to the gradient derivation
    # on page 4 of the Batch Normalization paper (2015, Ioffe and Szegedy)
    dx_roof = dout * gamma
    
    dvariance = dx_roof * (x - mean) * (-1.0 / 2.0) * (variance_pl_eps)**(-3.0/2.0)
    dvariance = np.sum(dvariance, axis=0)
    
    dmean = np.sum(dx_roof * (-inv_variance_pl_eps_sqrt), axis=0)
    dmean += dvariance * (np.sum(-2 * x_min_mean, axis=0)) / m
    
    dx = dx_roof * inv_variance_pl_eps_sqrt
    dx += dvariance * 2 * (x_min_mean) / m
    dx += dmean / m
    
    dbeta = np.sum(dout * 1.0, axis=0)
    dgamma = np.sum(dout * x_roof, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) > p) / (1.0 - p)
        out = mask * x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']
    p = dropout_param['p']
    
    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape   
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    # Assuming that all filters have the same size and also that all images have the same size
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride
    assert(H_out == int(H_out) and W_out == int(W_out))
    H_out, W_out = int(H_out), int(W_out)
    
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    out = np.zeros((N, F, H_out, W_out))
    
    # For a concrete image i, filter f, and output pixel (h_o, w_o), the value of this output pixel can be calculated
    # as a dot product of a certain number of image samples from the image i and the weights of the filter f (assuming
    # that we straighten them out into 1 dimension using ravel()). Note that as we change (h_o, w_o) but are still in the
    # same image i and filter f, the weights remain constant but the image samples change and they are given by a sliding
    # window looking into the image i.
    for i in range(N):
        for f in range(F):
            for h_o in range(H_out):
                for w_o in range(W_out):
                    # The sliding window position in the original image. Note that the window starts in the top left corner
                    # of the image. It then shifts right (the w_o index increases more frequently) taking stride-long steps
                    # and once it reaches the right edge of the image returns back to the left edge and takes a stride-long
                    # step down in the vertical direction, repeating the process.
                    top = h_o * stride
                    bottom = h_o * stride + HH
                    left = w_o * stride
                    right = w_o * stride + WW
                    window_samples = x_pad[i, :, top:bottom, left:right]
                    
                    out[i, f, h_o, w_o] = np.dot(window_samples.ravel(), w[f].ravel()) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_pad, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives. (N, F, H_out, W_out)
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x (N, C, H, W)
    - dw: Gradient with respect to w (F, C, HH, WW)
    - db: Gradient with respect to b (F,)
    """
    x_pad, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    N = x_pad.shape[0]   
    F, C, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    # We keep the padding on dx so that we can keep the same indexing scheme for x_pad as for dx
    dx, dw, db = np.zeros_like(x_pad), np.zeros_like(w), np.zeros_like(b)
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    # db
    # We sum over all channels, width and height but not over the axis 1 which indexes the kernels
    db = dout.sum(axis=(0, 2, 3)) 
    
    # dw
    # The partial derivative of loss w.r.t. a weight of a filter f is calculated as a sum over all the input images i
    # of all the pixels that were multiplied by this weight and thus contributed to the value
    # of an output pixel at (i, f, h_o, w_o). This is  multiplied by dl/d(out[i, f, h_o, w_o]) i.e., 
    # the partial derivative of the loss w.r.t a single output pixel.
    #
    # dx
    # The derivative of the loss w.r.t. an original pixel in an image i is the sum over all output pixels
    # in the output image i, over all filters f and of all the weights
    # in these filters that contributed to the value of the output pixel at (i, f, h_o, w_o). This is multiplied by 
    # dl/d(out[i, f, h_o, w_o]).
    for i in range(N):
        for h_o in range(H_out):
            for w_o in range(W_out):
                for f in range(F):
                    top = h_o * stride
                    bottom = h_o * stride + HH
                    left = w_o * stride
                    right = w_o * stride + WW
                    window_samples = x_pad[i, :, top:bottom, left:right]
                    
                    dloss_doutput_pixel = dout[i, f, h_o, w_o]
                    dw[f] += window_samples * dloss_doutput_pixel
                    dx[i, :, top:bottom, left:right] += w[f] * dloss_doutput_pixel
                    
    # Remove padding from dx
    dx = dx[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    W_out = int((W - pool_width) / stride + 1)
    H_out = int((H - pool_height) / stride + 1)
    out = np.zeros((N, C, H_out, W_out))
    
    for i in range(N):
        for h_o in range(H_out):
            for w_o in range(W_out):
                top = h_o * stride
                bottom = h_o * stride + pool_height
                left = w_o * stride
                right = w_o * stride + pool_width
                window_samples = x[i, :, top:bottom, left:right]
                
                out[i, :, h_o, w_o] = np.amax(window_samples, axis=(1, 2))
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives (N, C, H_out, W_out)
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x (N, C, H, W)
    """
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H_out, W_out = np.shape(dout)
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    dx = np.zeros_like(x) # (N, C, H, W)
    
    for i in range(N):
        for h_o in range(H_out):
            for w_o in range(W_out):
                top = h_o * stride
                bottom = h_o * stride + pool_height
                left = w_o * stride
                right = w_o * stride + pool_width
                window_samples = x[i, :, top:bottom, left:right]
                
                # Reshape so that we can get the indices of the max elements in the window for each channel
                window_samples = window_samples.reshape((C, pool_width * pool_height))
                maxes = np.argmax(window_samples, axis=1)
                
                # The local derivative for the max elements is 1, all other are 0 (they have no influence on the output)
                dout_din = np.zeros_like(window_samples)
                dout_din[np.arange(C), maxes] = 1.0
                
                # Reshape them back to window dimensions
                dout_din = dout_din.reshape((C, pool_height, pool_width))
                
                # Note that we want the dout derivative w.r.t the channels for this output pixel be broadcasted along
                # the dimensions 1 and 2 so we set them to 1
                dx[i, :, top:bottom, left:right] += dout_din * dout[i, :, h_o, w_o].reshape((C, 1, 1))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    
    # Reshape the input so that there are N*H*W examples of C dimensions (that is,
    # the data are looked upon as pixels of depth three without additional structure).
    N, C, H, W = x.shape
    x = x.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    
    # Reshape the normalized output back into the (N, C, H, W) format
    stride = H * W
    out_reshaped = np.zeros((N, C, H, W))
    for i in range(N):
        out_reshaped[i] = out[i*stride:(i+1)*stride].T.reshape(C, H, W)
    out = out_reshaped
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    
    # See comments for forward pass.
    N, C, H, W = dout.shape
    dout = dout.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    
    stride = H * W
    dx_reshaped = np.zeros((N, C, H, W))
    for i in range(N):
        dx_reshaped[i] = dx[i*stride:(i+1)*stride].T.reshape(C, H, W)
    dx = dx_reshaped
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
