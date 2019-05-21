import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .costs import *
from .activations import *
from scipy.signal import correlate as SC


class con:
    def __init__(self,inputs, filters,filtersize=5):
        outputs=filters
        self.filtersize=filtersize
        self.inputs=inputs
        self.outputs=outputs
        self.gw=np.zeros((inputs,outputs))
        self.filters=filters
        self.gb=np.zeros((1,outputs))
        self.mw=np.zeros((inputs,outputs))
        self.mb=np.zeros((1,outputs))
        self.delw=np.zeros((inputs,outputs))
        self.delb=np.zeros((1,outputs))
        self.weights=np.array([])
        self.biases=np.array([])
    def Act_derivative(self,Z):
        return Z>0

    def weightinit(self,seed=False):
        if seed!=False:
            np.random.seed(seed)

        self.weights=np.random.randn(self.filtersize,self.inputs,self.filters)*np.sqrt(2/(self.inputs+self.filters))
        self.biases=np.random.randn(1,self.filters)*np.sqrt(2/(self.inputs+self.filters))


    def forward(self,A,p=1):

        self.A=A
        if not self.weights.size:
            self.weightinit()
        padrows=int((self.weights.shape[0]-1)/2)

        padA=np.vstack((np.zeros((padrows,self.A.shape[1])),self.A,np.zeros((padrows,self.A.shape[1]))))
        self.H=np.empty((self.A.shape[0],self.filters))
        counter=0
        for k in range(self.filters):
            # print(self.weights[:,:,k].shape)
            self.H[:,k]=SC(padA,self.weights[:,:,k], mode="valid").reshape(-1)

        self.H+=self.biases
        self.Z=ReLU(self.H)
        return self.Z
        # self.A=np.vstack((np.zeros((2,A.shape[1])),A,np.zeros((2,A.shape[1]))))
        # print(self.A.shape)
        # self.Y=np.empty((self.A.shape[0]-4,self.fils))
        # for i in range(self.filts[:,:,].shape[2]):
        #     print(self.Y[:,i].shape)
        #     self.Y[:,i]=(SC(self.A,self.filts[:,:,i], mode='valid')+self.biases[i])
        # self.Z=ReLU(self.Y)
        # return self.Z

    def forwardnest(self,A,mu,p=1):
        self.A=A
        self.weights=np.random.randn(self.filtersize,self.inputs,self.filters)*np.sqrt(2/(self.inputs+self.filters))
        self.biases=np.random.randn(1,self.filters)*np.sqrt(2/(self.inputs+self.filters))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padA=np.vstack((np.zeros((padrows,self.A.shape[1])),self.A,np.zeros((padrows,self.A.shape[1]))))
        self.H=np.empty((self.A.shape[0],self.filters))
        for k in range(self.filters):
            self.H[:,k]=SC(padA,self.weights[:,:,k], mode="valid").reshape(-1)
        self.H+=self.biases
        self.Z=ReLU(self.H)
        return self.Z

    def backward(self,D,lr,l1,l2):

        derivterm=D*self.Act_derivative(self.Z)
        Dout=np.empty((D.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            Dout[:,k]=SC(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=lr*SC(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=lr*(np.sum(derivterm, axis=0))
        return Dout

    def backwardnest(self, D, lr, mu,l1,l2):
        derivterm=D*self.Act_derivative(self.Z)
        Dout=np.empty((D.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            Dout[:,k]=SC(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=lr*SC(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=lr*(np.sum(derivterm, axis=0))
        return Dout
    def backwardada(self,D,lr,mu,nest,l1,l2):
        derivterm=D*self.Act_derivative(self.Z)
        Dout=np.empty((D.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            Dout[:,k]=SC(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=lr*SC(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=lr*(np.sum(derivterm, axis=0))
        return Dout
    def backwardadanest(self, D, lr, mu,l1,l2):
        derivterm=D*self.Act_derivative(self.Z)
        Dout=np.empty((D.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            Dout[:,k]=SC(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=lr*SC(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=lr*(np.sum(derivterm, axis=0))
        return Dout
    def backwardrms(self,D,lr,gamma,mu,nest,l1,l2):
        derivterm=D*self.Act_derivative(self.Z)
        Dout=np.empty((D.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            Dout[:,k]=SC(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=lr*SC(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=lr*(np.sum(derivterm, axis=0))
        return Dout
    def backwardrmsnest(self, D, lr, gamma, mu,l1,l2):
        derivterm=D*self.Act_derivative(self.Z)
        Dout=np.empty((D.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            Dout[:,k]=SC(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=lr*SC(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=lr*(np.sum(derivterm, axis=0))
        return Dout

    def backwardadam(self,D,lr,gamma,mu,count,l1,l2):
        derivterm=D*self.Act_derivative(self.Z)
        Dout=np.empty((D.shape[0],self.A.shape[1]))
        padrows=int(np.ceil(((self.weights.shape[0]-1)/2)))
        padder=np.vstack((np.zeros((padrows,derivterm.shape[1])),derivterm,np.zeros((padrows,derivterm.shape[1]))))
        for k in range(self.weights[0,:,0].shape[0]):
            Dout[:,k]=SC(padder,self.weights[::-1,k,:],mode='valid').reshape(-1)
            for l in range(self.weights[0,0,:].shape[0]):
                self.weights[:,k,l]-=lr*SC(self.A[:,k],derivterm[:,l], mode="valid")
        self.biases-=lr*(np.sum(derivterm, axis=0))
        return Dout
