import numpy as np
def Sigmoid (x):
    return(1/(1+np.exp(-x)))
def DerivSig(x):
    return np.multiply(1/(1+np.exp(-x)),(1-(1/(1+np.exp(-x)))))
def Softmax(h):

    return np.exp(h)/np.sum(np.exp(h),axis=1).reshape(len(h[:,0]),1)
def Tanh(h):
    return np.tanh(h)
def DerivTanh(a):
    return 1-np.multiply(a,a)
def ReLU(x):
    return np.multiply(x, (x > 0))
def DerivRelu(x):
    return x>0
def leakRelu(x):
    return np.multiply(x, (x > 0))+.2*np.multiply(x,(x<=0))
def leakDerivRelu(x):
    return x>0 +.2*(x<=0)
def Prelu(X,p):
    z=X*(X>0) +p*X*(X<=0)
    return(z)
def DerivPreluP(a):
    return a*(a<=0)
def DerivPreluA(X,p):
    return (X>0) +p*(X<=0)
