import numpy as np
from random import shuffle
from past.builtins import xrange

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
  train_num, train_dimention = X.shape
  class_num = np.max(y) + 1
  for i in range(train_num):
    score = X[i].dot(W)
    score -= np.max(score) #改变值得最大值为0，这样指数的结果不会超过1，否则一个大的数的指数可能超过了最大值
    score_exp = np.exp(score)
    score_property = score_exp / np.sum(score_exp)
    loss += - np.log( score_property[y[i]] )
    for j in range(class_num):
      if j == y[i]:
        dW[:, j] += (score_property[j] - 1) * X[i]
      else:
        dW[:, j] += score_property[j] * X[i]
    
  loss /= train_num
  loss += 0.5 * reg * np.sum(W * W)
  dW /= train_num
  dW += reg * W 
  pass
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
  train_num, train_class = X.shape
  scores = np.dot(X, W) #(N, C)
  scores_max = np.max(scores, axis=1)
  scores_max = np.reshape(scores_max, [-1, 1])
  scores = scores - scores_max #将每一行数据减去该行最大值，使其值均小于0，防止指数最大值超过限制
  scores_exp = np.exp(scores) #(N, C)
  scores_sum = np.sum(scores_exp, axis=1) #(N, )
  scores_sum = np.reshape(scores_sum, [-1, 1]) #(N, 1)
  scores_property = scores_exp / scores_sum #(N, C)
  
  scores_loss = - np.log( scores_property[np.arange(train_num), y] )
  loss += np.sum(scores_loss)
  loss /= train_num
  loss += 0.5 * reg * np.sum(W * W)
    
  scores_property[np.arange(train_num), y] -= 1
  dW = (X.T).dot(scores_property)
  dW = dW/train_num + reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

