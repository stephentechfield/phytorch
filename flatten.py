import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .costs import *
from .activations import *
from scipy.signal import correlate as SC

class flatten:
    def __init__(self,inputs,outputs):
        pass
    def forward(self,A,p=1):
        self.dimentions=A.shape
        self.A=A
        U=A.flatten().reshape(1,-1)
        return U


    def forwardnest(self,A,mu,p=1):
        self.dimentions=A.shape

        return A.flatten().reshape(1,-1)

    def backward(self,D,lr,l1,l2):
        self.Dout=D
        D=self.Dout.reshape(*self.dimentions)
        return D
    def backwardnest(self, D, lr, mu,l1,l2):
        self.Dout=D
        self.Dout.reshape(*self.dimentions)
        return self.Dout
    def backwardada(self,D,lr,mu,nest,l1,l2):
        self.Dout=D
        self.Dout.reshape(*self.dimentions)
        return self.Dout
    def backwardadanest(self, D, lr, mu,l1,l2):
        self.Dout=D
        self.Dout.reshape(*self.dimentions)
        return self.Dout
    def backwardrms(self,D,lr,gamma,mu,nest,l1,l2):
        self.Dout=D
        self.Dout.reshape(*self.dimentions)
        return self.Dout
    def backwardrmsnest(self, D, lr, gamma, mu,l1,l2):
        self.Dout=D
        self.Dout.reshape(*self.dimentions)
        return self.Dout

    def backwardadam(self,D,lr,gamma,mu,count,l1,l2):
        self.Dout=D
        self.Dout.reshape(*self.dimentions)
        return self.Dout
