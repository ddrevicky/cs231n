import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  num_dim = X.shape[1]
  num_classes = W.shape[1]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  for i in range(num_train):
    s = X[i].dot(W)
    exp_s = np.exp(s - np.max(s))
    sum_exp_s = np.sum(exp_s)
    
    # Note: The gradient computation is derived from the (modified) formula
    # L_i = f_yi + log(sum_over_j(e^{f_j})).
    for j in range(num_classes):
        if j == y[i]:
            dW[:,j] += X[i].T * (-1.0 + exp_s[j] / sum_exp_s)
        else:
            dW[:,j] += X[i].T * (exp_s[j] / sum_exp_s)
    
    # Commented code is computing gradient for the formula L_i = -log( e^{f_yi} / sum_over_j(e^{f_j}) ).
    # The gradients are of course identical since the formulas are equal.
    #dp = -1.0 / p
    #ds = (-exp_s[y[i]] * exp_s) / den**2
    #ds[y[i]] = (den*exp_s[y[i]] - exp_s[y[i]]**2) / den**2
    #ds *= dp
    #dW += X[i].reshape(num_dim, 1) * ds
    
    loss += -np.log(exp_s[y[i]] / sum_exp_s)
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
    
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train = X.shape[0]
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  S = X.dot(W) 
  S -= np.max(S, axis = 1).reshape((num_train, 1))
  exp_S = np.exp(S)
  sum_exp_S = np.sum(exp_S, axis = 1)
  
  dLdS = exp_S / sum_exp_S.reshape((num_train, 1))
  dLdS[np.arange(num_train), y] += -1.0
  dW = X.T.dot(dLdS) / num_train
  dW += 2 * reg * W
    
  loss = exp_S[np.arange(num_train), y] / sum_exp_S
  loss = -np.log(loss)
  loss = loss.sum() / num_train
  loss += reg * np.sum(W * W)
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

