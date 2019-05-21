import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .costs import *
from .activations import *

class sigmoidO:
    def __init__(self,inputs,outputs):
        self.inputs=inputs
        self.outputs=outputs
        self.gw=np.zeros((inputs,outputs))
        self.gb=np.zeros((1,outputs))
        self.mw=np.zeros((inputs,outputs))
        self.mb=np.zeros((1,outputs))
        self.delw=np.zeros((inputs,outputs))
        self.delb=np.zeros((1,outputs))
        self.weights=np.random.randn(inputs,outputs)
        self.biases=np.random.randn(1,outputs)
    def forward(self,A,p=1):
        self.A=A*((np.random.rand(A.shape[0],A.shape[1])<p)/p)
        G=self.A@self.weights
        self.H=G+self.biases
        self.Z=Sigmoid(self.H)
        return self.Z
    def forwardnest(self,A,mu,p=1):
        self.A=A*((np.random.rand(A.shape[0],A.shape[1])<p)/p)
        self.weights+=mu*self.delw
        self.biases+=mu*self.delb
        self.G=self.A@self.weights
        self.H=self.G+self.biases
        self.Z=Sigmoid(self.H)
        return self.Z
    def backward(self,D,lr,l1,l2):
        self.D=D
        self.weights=self.weights-lr*(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))
        self.biases=self.biases-lr*(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))
        self.Dout=(self.Z-self.D)@self.weights.T
        return self.Dout
    def backwardnest(self, D, lr, mu,l1,l2):
        self.D=D
        ### remembering where we went and taking a step back
        self.gradw=lr*(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))
        self.weights=self.weights-mu*self.delw
        self.gradb=lr*(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))
        self.biases=self.biases-mu*self.delb
        ## moving forward with foresight
        self.delw=mu*self.delw-self.gradw
        self.weights+=self.delw
        self.delb=mu*self.delb-self.gradb
        self.biases+=self.delb
        self.Dout=(self.Z-self.D)@self.weights.T
        return self.Dout
    def backwardada(self,D,lr,mu,nest,l1,l2):
        self.D=D
        self.gw=self.gw+(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))**2
        self.gb=self.gb+(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))**2
        self.etaw=lr/(np.sqrt(self.gw+.000000001))
        self.etab=lr/(np.sqrt(self.gb+.000000001))
        self.Dout=(self.Z-self.D)@self.weights.T
        if not nest:
            self.mw=mu*self.mw-self.etaw*(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))
            self.mb=mu*self.mb-self.etab*(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))
            self.weights=self.weights+self.mw
            self.biases=self.biases+self.mb
        return self.Dout
    def backwardadanest(self, D, lr, mu,l1,l2):
        self.D=D
        ### remembering where we went and taking a step back
        self.gradw=self.etaw*(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))
        self.weights=self.weights-mu*self.delw
        self.gradb=self.etab*(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))
        self.biases=self.biases-mu*self.delb
        ## moving forward with foresight
        self.delw=mu*self.delw-self.gradw
        self.weights+=self.delw
        self.delb=mu*self.delb-self.gradb
        self.biases+=self.delb
        self.Dout=(self.Z-self.D)@self.weights.T
        return self.Dout
    def backwardrms(self,D,lr,gamma,mu,nest,l1,l2):
        self.D=D
        self.gw=(gamma*self.gw)+np.multiply((1-gamma),(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))**2)
        self.gb=(gamma*self.gb)+np.multiply((1-gamma),(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))**2)
        self.etaw=lr/(np.sqrt(self.gw+.000000001))
        self.etab=lr/(np.sqrt(self.gb+.000000001))
        self.Dout=(self.Z-self.D)@self.weights.T
        if not nest:
            self.mw=mu*self.mw-self.etaw*(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))
            self.mb=mu*self.mb-self.etab*(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))
            self.weights=self.weights+self.mw
            self.biases=self.biases+self.mb
        return self.Dout
    def backwardrmsnest(self, D, lr, gamma, mu,l1,l2):
        self.D=D
        ### remembering where we went and taking a step back
        self.gradw=self.etaw*(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))
        self.weights=self.weights-mu*self.delw
        self.gradb=self.etab*(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))
        self.biases=self.biases-mu*self.delb
        ## moving forward with foresight
        self.delw=mu*self.delw-self.gradw
        self.weights+=self.delw
        self.delb=mu*self.delb-self.gradb
        self.biases+=self.delb
        self.Dout=(self.Z-self.D)@self.weights.T
        return self.Dout
    def backwardadam(self,D,lr,gamma,mu,count,l1,l2):
        self.D=D
        self.mw=mu*self.mw+(1-mu)*(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))
        self.mb=mu*self.mb+(1-mu)*((np.sum(self.Z-self.D,axis=0))+l1*np.sign(self.biases)+l2*(self.biases))
        self.gw=(gamma*self.gw)+np.multiply((1-gamma),(self.A.T@(self.Z-self.D)+l1*np.sign(self.weights)+l2*(self.weights))**2)
        self.gb=(gamma*self.gb)+np.multiply((1-gamma),(np.sum(self.Z-self.D,axis=0)+l1*np.sign(self.biases)+l2*(self.biases))**2)
        self.mhatw=self.mw/(1+mu**count)
        self.mhatb=self.mb/(1+mu**count)
        self.vhatw=self.gw/(1+gamma)
        self.vhatb=self.gb/(1+gamma)
        self.etaw=lr/(np.sqrt(self.vhatw+.000000001))
        self.etab=lr/(np.sqrt(self.vhatb+.000000001))
        self.weights=self.weights-self.etaw*self.mhatw
        self.biases=self.biases-self.etab*self.mhatb
        self.Dout=(self.Z-self.D)@self.weights.T
        return self.Dout
