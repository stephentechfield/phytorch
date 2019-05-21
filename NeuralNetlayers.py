import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .costs import *
from .activations import *
from .PerformanceIndicators import *
from .prelu import *
from .relu import *
from .tanh import *
from .softmax import *
from .sigmoid import *
from .ident import *
from .sigmoidO import *
from .pool import *
from .spp import *
from .flatten import *
from .con import *


class NeuralNetL(object):

    def __init__(self,indims, nodes, activations, optimizer=False, nest=False):
        self.optimizer=optimizer
        self.nest=nest
        self.indims=indims
        self.nodes=[self.indims]+nodes
        self.layers=[]
        self.activations=activations
        for i in range (0,len(nodes)):
            if activations[i]=="tanh":
                self.layers.append(tanh(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="softmax":
                self.layers.append(softmax(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="relu":
                self.layers.append(relu(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="prelu":
                self.layers.append(prelu(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="sigmoid":
                self.layers.append(sigmoid(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="sigmoidO":
                self.layers.append(sigmoidO(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="ident":
                self.layers.append(ident(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="con":
                self.layers.append(con(inputs= self.nodes[i],filters = self.nodes[i+1]))
            if activations[i]=="pool":
                self.layers.append(pol(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="spp":
                self.layers.append(spp(inputs= self.nodes[i],outputs = self.nodes[i+1]))
            if activations[i]=="flatten":
                self.layers.append(flatten(inputs= self.nodes[i],outputs = self.nodes[i+1]))
    def predict(self,X,y,lr, mu, gamma,l1=0,l2=0,p=1):
        temp=X.copy()
        if self.nest==False:
            for i in range(0,len(self.layers)):
                temp=self.layers[i].forward(temp,p)
            self.prediction=temp
            return self.prediction
        elif self.nest==True:
            if self.optimizer == False:
                for i in range(0,len(self.layers)):
                    temp=self.layers[i].forwardnest(temp,mu,p)
                self.prediction=temp
                return self.prediction
            elif self.optimizer == "ada":
                for i in range(0,len(self.layers)):
                    temp=self.layers[i].forward(temp,p)
                temp=y.copy()
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardada(temp,lr,mu,self.nest,l1,l2)
                temp=X.copy()
                for i in range(0,len(self.layers)):
                    temp=self.layers[i].forwardnest(temp,mu,p)
                self.prediction=temp
                return self.prediction
            elif self.optimizer == "rms":
                for i in range(0,len(self.layers)):
                    temp=self.layers[i].forward(temp,p)
                temp=y.copy()
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardrms(temp,lr,mu, gamma, self.nest,l1,l2)
                temp=X.copy()
                for i in range(0,len(self.layers)):
                    temp=self.layers[i].forwardnest(temp,mu,p)
                self.prediction=temp
                return self.prediction
            else:
                for i in range(0,len(self.layers)):
                    temp=self.layers[i].forwardnest(temp,mu,p)
                self.prediction=temp
                return self.prediction

        elif self.nest=="nest":
            for i in range(0,len(self.layers)):
                temp=self.layers[i].forwardnest(temp,mu,p)
            self.prediction=temp
            return self.prediction
        else:
            for i in range(0,len(self.layers)):
                temp=self.layers[i].forward(temp,p)
            self.prediction=temp
            return self.prediction

    def BackProp(self,X,lr,gamma,mu,count,l1,l2):
        temp=X.copy()
        if self.nest==False:
            if self.optimizer ==False:
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backward(temp,lr,l1,l2)
            elif self.optimizer == "ada":
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardada(temp,lr,mu,self.nest,l1,l2)
            elif self.optimizer == "rms":
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardrms(temp,lr,gamma,mu, self.nest,l1,l2)
            elif self.optimizer == "adam":
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardadam(temp,lr,gamma,mu,count,l1,l2)
            else:
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backward(temp,lr,l1,l2)
        else:
            if self.optimizer ==False:
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardnest(temp,lr,mu,l1,l2)
            elif self.optimizer == "ada":
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardadanest(temp,lr,mu,l1,l2)
            elif self.optimizer == "rms":
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardrmsnest(temp,lr,gamma,mu,l1,l2)
            elif self.optimizer == "adam":
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardadam(temp,lr,gamma,mu,count,l1,l2)
            else:
                for i in range(len(self.layers)-1,-1,-1):
                    temp=self.layers[i].backwardnest(temp,lr,l1,l2)


    def train(self, X, y, Xval=[], yval=[], epochs=100, p=1,lr=.0001, l1=0,l2=0,T=80, Sch=False, Inv=False, decay=0.9, Expen=False, gamma=0.9,mu=0.0,batchsize=[]):#, mu=0, T=80, Sch=False, Inv=False, decay=0.9, Expen=False,  l1=0,l2=0, Xval=[],yval=[]):
        self.cost=[]
        if Xval==[]:
            Xval=X
            yval=y

        if batchsize==[]:
            batchsize=len(X)
            numbatch=1
        else:
            numbatch=int(np.ceil(len(X)/batchsize))
        count=0
        costold=200000
        iteration=0
        acc=0
        for i in range(epochs):
            lrs=lr

            if Sch==True:
                if iteration>T:
                    lr=lrs*decay
            if Inv==True:
                if iteration>1:
                    lr=lr/((decay*iteration)+1)
            if Expen==True:
                if iteration>1:
                    lr=lrs*np.exp(decay*iteration)
            iteration=iteration+1

            for j in range(0,numbatch):
                newX=X[j*(batchsize):(j+1)*batchsize,].copy()
                newy=y[j*(batchsize):(j+1)*batchsize,].copy()
                count=count+1
                phat=self.predict(newX,newy,lr,mu,gamma)
                self.BackProp(newy,lr,gamma,mu,count,l1,l2)

                if self.activations[-1]=='ident':
                    costs=SSE(yval,self.predict(Xval,yval,lr,mu,gamma,l1,l2))
                    self.cost.append(costs)

                elif y.shape[1]>1:
                    costs=GCEC(yval,self.predict(Xval,yval,lr,mu,gamma,l1,l2))
                    self.cost.append(costs)
                else:
                    costs =BCEC(yval,self.predict(Xval,yval,lr,mu,gamma,l1,l2))
                    self.cost.append(costs)
            iteration+=1
        plt.plot(self.cost)
