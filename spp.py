import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .costs import *
from .activations import *
from scipy.signal import correlate as SC

class spp:
    def __init__(self,inputs,outputs):
        self.inputs=inputs
        self.outputs=outputs
        self.gw=np.zeros((inputs,outputs))
        self.filts={}

        self.gb=np.zeros((1,outputs))
        self.mw=np.zeros((inputs,outputs))
        self.mb=np.zeros((1,outputs))
        self.delw=np.zeros((inputs,outputs))
        self.delb=np.zeros((1,outputs))

    def forward(self,A,p=1):
        self.dims=A.shape
        o1=int(A.shape[1])
        out4 = np.empty((4,o1))
        locations4 = np.empty((4,o1))
        cX=0
        s=int(A.shape[0]/4)
        for i in range(0,A.shape[0],s):
            out4[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations4[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1

        out2 = np.empty((2,o1))
        locations2 = np.empty((2,o1))
        cX=0
        s=int(A.shape[0]/2)
        for i in range(0,A.shape[0],s):
            out2[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations2[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1

        out1 = np.max(A, axis=0)
        location1=np.argmax(A, axis=0)

        self.locations=np.vstack((locations4,locations2,location1)).astype(int)
        sforward=np.vstack((out4,out2,out1))

        return sforward

    def forwardnest(self,A,mu,p=1):
        self.dims=A.shape
        o1=int(A.shape[1])
        out4 = np.empty((4,o1))
        locations4 = np.empty((4,o1))
        cX=0
        s=int(A.shape[0]/4)
        for i in range(0,A.shape[0],s):
            out4[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations4[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1

        out2 = np.empty((2,o1))
        locations2 = np.empty((2,o1))
        cX=0
        s=int(A.shape[0]/2)
        for i in range(0,A.shape[0],s):
            out2[cX,:] = np.max(A[i:i+s,:], axis=0)
            locations2[cX,:]=np.argmax(A[i:i+s,:], axis=0)
            cX+=1

        out1 = np.max(A, axis=0)
        location1=np.argmax(A, axis=0)

        self.locations=np.vstack((locations4.astype(int),locations2.astype(int),location1.astype(int)))
        return np.vstack((out4,out2,out))

    def backward(self,D,lr,l1,l2):
        Dout = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                Dout[i:i+s,j][self.locations[cX,j]] = D[cX,j]
            cX+=1
        return Dout



    def backwardnest(self, D, lr, mu,l1,l2):
        Dout = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                Dout[i:i+s,j][self.locations[cX,j]] = D[cX,j]
            cX+=1
        return Dout
    def backwardada(self,D,lr,mu,nest,l1,l2):
        Dout = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                Dout[i:i+s,j][self.locations[cX,j]] = D[cX,j]
            cX+=1
        return Dout
    def backwardadanest(self, D, lr, mu,l1,l2):
        Dout = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                Dout[i:i+s,j][self.locations[cX,j]] = D[cX,j]
            cX+=1
        return Dout
    def backwardrms(self,D,lr,gamma,mu,nest,l1,l2):
        Dout = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                Dout[i:i+s,j][self.locations[cX,j]] = D[cX,j]
            cX+=1
        return Dout
    def backwardrmsnest(self, D, lr, gamma, mu,l1,l2):
        Dout = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                Dout[i:i+s,j][self.locations[cX,j]] = D[cX,j]
            cX+=1
        return Dout

    def backwardadam(self,D,lr,gamma,mu,count,l1,l2):
        Dout = np.zeros(self.dims)
        s=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],s):
            for j in range(self.dims[1]):
                Dout[i:i+s,j][self.locations[cX,j]] = D[cX,j]
            cX+=1
        return Dout
