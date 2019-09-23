import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim = X.shape[1]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    margins = scores - scores[y[i]] + 1.0
    margins[y[i]] = 0.0
    margins = np.maximum(margins, 0)
    loss += margins.sum()

    dmargins = 1.0 * (margins > 0.0).astype(int)
    dscores = np.ones_like(scores) * dmargins
    dscores[y[i]] = np.sum(-1.0 * dmargins)
    dW += X[i].reshape(num_dim, 1) * dscores

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim = X.shape[1] 

  scores = X.dot(W) # scores: NxC
  subtract_mask = scores[np.arange(num_train), y].reshape((num_train,1))
  margins = scores - subtract_mask + 1
  margins[np.arange(num_train), y] = 0
  margins = np.maximum(0, margins)

  loss = margins.sum() / num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dmargins = 1.0 * (margins > 0.0).astype(int)
  dscores = np.ones_like(scores) * dmargins
  dscores[np.arange(num_train), y] = np.sum(-1.0 * dmargins, axis = 1)
  dW = X.T.dot(dscores) / num_train
  dW += reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
