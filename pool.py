import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .costs import *
from .activations import *
from scipy.signal import correlate as SC

class pool:
    def __init__(self,inputs,outputs, fils=32):
        self.inputs=inputs
        self.outputs=outputs
        self.fils=fils
        self.gw=np.zeros((inputs,outputs))
        self.filts={}
        for i in range(self.fils):
            self.filts[i]=(np.random.rand(5,inputs))
        self.gb=np.zeros((1,outputs))
        self.mw=np.zeros((inputs,outputs))
        self.mb=np.zeros((1,outputs))
        self.delw=np.zeros((inputs,outputs))
        self.delb=np.zeros((1,outputs))
        self.weights=np.random.randn(inputs,outputs)
        self.biases=np.random.randn(fils)
    def forward(A,p=1):
        o0=int(A.shape[0]/2)
        o1=int(A.shape[1])

        Z = np.empty((o0,o1))
        self.locations = np.empty((o0,o1))
        cX=0
        for i in range(0,A.shape[0],2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.locations[cX,:]=np.argmax(A.shape[i:i+2,:], axis=0)
            cX+=1
        return Z

    def forwardnest(self,A,mu,p=1):
        o0=int(A.shape[0]/2)
        o1=int(A.shape[1])

        Z = np.empty((o0,o1))
        self.locations = np.empty((o0,o1))
        cX=0
        for i in range(0,A.shape[0],2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.locations[cX,:]=np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z


    def backward(self,D,lr,l1,l2):
        self.D=D
        o0=int(self.D.shape[0]*2)
        o1=int(self.D.shape[1])
        self.Dout = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                self.Dout[i:i+2,j][self.locations[cX,j]] = self.D[cX,j]
            cX+=1
        return self.Dout


    def backward(self,D,lr,l1,l2):
        self.D=D
        self.weights=self.weights-lr*(self.A.T@(self.D*DerivRelu(self.Z))+l1*np.sign(self.weights)+l2*(self.weights))
        self.biases=self.biases-lr*(np.sum(self.D*DerivRelu(self.Z),axis=0)+l1*np.sign(self.biases)+l2*(self.biases))
        self.Dout=(self.D*DerivRelu(self.Z))@self.weights.T
        return self.Dout

    def backwardnest(self, D, lr, mu,l1,l2):
        self.D=D
        o0=int(self.D.shape[0]*2)
        o1=int(self.D.shape[1])
        self.Dout = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                self.Dout[i:i+2,j][self.locations[cX,j]] = self.D[cX,j]
            cX+=1
        return self.Dout
    def backwardada(self,D,lr,mu,nest,l1,l2):
        self.D=D
        o0=int(self.D.shape[0]*2)
        o1=int(self.D.shape[1])
        self.Dout = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                self.Dout[i:i+2,j][self.locations[cX,j]] = self.D[cX,j]
            cX+=1
        return self.Dout
    def backwardadanest(self, D, lr, mu,l1,l2):
        self.D=D
        o0=int(self.D.shape[0]*2)
        o1=int(self.D.shape[1])
        self.Dout = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                self.Dout[i:i+2,j][self.locations[cX,j]] = self.D[cX,j]
            cX+=1
        return self.Dout
    def backwardrms(self,D,lr,gamma,mu,nest,l1,l2):
        self.D=D
        o0=int(self.D.shape[0]*2)
        o1=int(self.D.shape[1])
        self.Dout = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                self.Dout[i:i+2,j][self.locations[cX,j]] = self.D[cX,j]
            cX+=1
        return self.Dout
    def backwardrmsnest(self, D, lr, gamma, mu,l1,l2):
        self.D=D
        o0=int(self.D.shape[0]*2)
        o1=int(self.D.shape[1])
        self.Dout = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                self.Dout[i:i+2,j][self.locations[cX,j]] = self.D[cX,j]
            cX+=1
        return self.Dout

    def backwardadam(self,D,lr,gamma,mu,count,l1,l2):
        self.D=D
        o0=int(self.D.shape[0]*2)
        o1=int(self.D.shape[1])
        self.Dout = np.zeros((o0,o1))
        cX=0
        for i in range(0,o0,2):
            for j in range(o1):
                self.Dout[i:i+2,j][self.locations[cX,j]] = self.D[cX,j]
            cX+=1
        return self.Dout
