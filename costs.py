import numpy as np
def GCEC(y, phat):
    return -np.sum(np.multiply(y,np.log(phat)))/y.shape[0]
def BCEC(y, yhat):
    zeros=np.where(y==0)
    ones=np.where(y==1)
    Err = np.hstack((-np.log(yhat[ones]),-np.log(1-yhat[zeros])))
    return np.mean(Err)
def fronorm(y,yhat):
    return np.trace((y-yhat).T@(y-yhat))/len(y[:,0])
def SSE(y,yhat):
    return np.sum(np.multiply(y-yhat,y-yhat))/len(y[:,0])
