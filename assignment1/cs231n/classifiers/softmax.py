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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  N = len(scores)

  exp_scores = np.exp(scores-np.max(scores, axis=1, keepdims=True))
  probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  # correct_logprobs = -probs[range(N),y] * np.sum(np.log(probs), axis=1)
  correct_logprobs = np.zeros(N)
  for i in range(N):
    correct_logprobs[i] = -np.log(probs[i][y[i]])
  data_loss = np.sum(correct_logprobs)/N
  reg_loss = 0.5 * reg * np.sum(W*W)
  loss = data_loss + reg_loss

  delta3 = np.zeros_like(probs)
  for i in range(N):
    delta3[i][y[i]] -= 1
  delta3 += probs
  dW = X.T.dot(delta3) / N + reg * W
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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  N = len(scores)

  exp_scores = np.exp(scores-np.max(scores, axis=1, keepdims=True))
  probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  # correct_logprobs = -probs[range(N),y] * np.sum(np.log(probs), axis=1)
  correct_logprobs = -np.log(probs[range(N),y])
  data_loss = np.sum(correct_logprobs)/N
  reg_loss = 0.5 * reg * np.sum(W*W)
  loss = data_loss + reg_loss

  delta3 = np.zeros_like(probs)
  delta3[np.arange(N), y] -= 1
  delta3 += probs
  dW = X.T.dot(delta3) / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

