import numpy as np
def accuracy(y, yhat):
    return np.mean(yhat==y)
def recall(y,yhat):
    return np.sum(np.multiply(yhat,y),axis=0)/np.sum(y,axis = 0)
