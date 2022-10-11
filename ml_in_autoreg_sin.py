# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:07:30 2022

@author: Student
"""

import torch
import torch.nn as nn 
import scipy as sc
import numpy as np
import sympy as sym
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sympy import init_printing
init_printing()

import matplotlib.pyplot as plt 
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# =============================================================================
# response generation
# =============================================================================
#y0 = [0.5]

t = np.linspace(0, 2, 40)
t_test=np.linspace(0, 2, 40)
delta_t=t[1]-t[0]

state=np.sin(10*t)
from scipy.integrate import odeint 
def model(y,t):
    dydt = np.sin(10*t)
    return dydt

# initial condition
total_ic=2
test_ic=1
y0 = torch.rand(total_ic,1)
y0_train=y0[0:total_ic-test_ic,:]

y0_test=y0[0:test_ic,:]
y_train=torch.zeros(len(y0_train),len(t))
y_test=torch.zeros(len(y0_test),len(t_test))
for i in range(len(y0_train)):

# solve ODE
    y = odeint(model,y0_train[i],t)
    y_ten=torch.tensor(y)
    for j in range(len(y_ten)):
        y_train[i,j]=y_ten[j,:]

    plt.figure(1)
    plt.plot(t,y) 
#true_sol=np.cos(10*t)

for i in range(len(y0_test)):
    print(i)

# solve ODE
    y_t = odeint(model,y0_test[i],t_test)
    y_test_ten=torch.tensor(y_t)
    for j in range(len(y_test_ten)):
        print(j)
        y_test[i,j]=y_test_ten[j,:]

    plt.figure(2)
    plt.plot(t_test,y_t)    
# y_test = odeint(model,y0_test,t)
# plt.figure(2)
# plt.plot(t,y_test)
# y_test=torch.tensor(y_test).reshape()
#y0 = [1.0, 1.0]
t=torch.tensor(t)

# =============================================================================
# 
# =============================================================================
time_steps=len(t)
delta_t=t[1]-t[0]
#y_0=torch.tensor(y0[0]).reshape(1,1)

#output_signal=torch.tensor(y)
#output_signal=output_signal.reshape(len(output_signal),1)

class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size_i,hidden_size_h0,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4,output_size):
        super(RNN, self).__init__()
        
        self.hidden_size_i = hidden_size_i
        self.hidden_size_h0 = hidden_size_h0
        #self.hidden_size2 = hidden_size2
        #self.hidden_size3 = hidden_size3
        #self.hidden_size4 = hidden_size4
        self.i2h1 = nn.Linear(input_size, hidden_size_i)
        self.h12h2 = nn.Linear(hidden_size_i, hidden_size_1)
        self.h22h3 = nn.Linear(hidden_size_1, hidden_size_2)
        self.h32h4 = nn.Linear(hidden_size_2, hidden_size_3)
        self.h42h5 = nn.Linear(hidden_size_3, hidden_size_4)
        self.h02h1 = nn.Linear(hidden_size_h0, hidden_size_i)
        #self.h12h2=  nn.Linear(hidden_size1, hidden_size2)
        #self.h22h3=  nn.Linear(hidden_size2, hidden_size3)
        #.h32h4=  nn.Linear(hidden_size3, hidden_size4)
        self.add2o=nn.Linear(hidden_size_h0,output_size)
        self.add2h=nn.Linear(hidden_size_h0,hidden_size_h0)
        #self.i2o = nn.Linear(input_size + hidden_size0, output_size)
        #self.h2o=nn.Linear(hidden_size1, output_size)
        
        self.tanh = nn.Tanh()
        
    def forward(self, input_tensor):
        #print("it",input_tensor.shape,"ht",hidden_tensor.shape)
        #combined = torch.cat((input_tensor, hidden_tensor), 1)
        
       # print("combined",combined.shape)
        
        hidden_i = self.i2h1(input_tensor)
        hidden_i = self.tanh(hidden_i)
        hidden_i=self.h12h2(hidden_i)
        hidden_i = self.tanh(hidden_i)
        hidden_i=self.h22h3(hidden_i)
        hidden_i = self.tanh(hidden_i)
        hidden_i=self.h32h4(hidden_i)
        hidden_i = self.tanh(hidden_i)
        hidden_i=self.h42h5(hidden_i)
        hidden_i = self.tanh(hidden_i)
        
        output=self.h12h2(hidden_i)
        output = self.tanh(output)
        output=self.h22h3(output)
        output = self.tanh(output)
        output=self.h32h4(output)
        output = self.tanh(output)
        output=self.h42h5(output)
        output = self.tanh(output)
        output=self.add2o(output)
       
        # hidden_h0 = self.h02h1(hidden_tensor)
        # hidden_h0 = self.tanh(hidden_h0)
        # hidden_h0=self.h12h2(hidden_h0)
        # hidden_h0 = self.tanh(hidden_h0)
        # hidden_h0=self.h22h3(hidden_h0)
        # hidden_h0 = self.tanh(hidden_h0)
        # hidden_h0=self.h32h4(hidden_h0)
        # hidden_h0 = self.tanh(hidden_h0)
        # hidden_h0=self.h42h5(hidden_h0)
        # hidden_h0 = self.tanh(hidden_h0)
       
        #print("hidden vector",hidden.shape)
        # output = hidden_i+hidden_h0
        # output=self.h12h2(output)
        # output = self.tanh(output)
        # output=self.h22h3(output)
        # output = self.tanh(output)
        # output=self.h32h4(output)
        # output = self.tanh(output)
        # output=self.h42h5(output)
        # output = self.tanh(output)
       
        
       
        
      
        
       # print("output vector",output.shape)
        return output
    
    # def init_hidden(self):
    #     return torch.zeros(1, self.hidden_size_h0)
hidden_size_i=32    
hidden_size_h0 = 32
hidden_size_1 = 128
hidden_size_2 = 256
hidden_size_3 = 128
hidden_size_4=32
input_size=4
output_size=1

rnn = RNN(input_size, hidden_size_i,hidden_size_h0,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4, output_size)

# =============================================================================
# # one step
# input_tensor=input_signal[:,0]
# input_tensor=input_tensor.reshape(1,1)
# hidden_tensor = rnn.init_hidden()
# 
# output, next_hidden = rnn(input_tensor, hidden_tensor)
# print(output.size())
# print(next_hidden.size())
# =============================================================================


#out=torch.zeros(input_tensor.size()[1],1)
learning_rate = 0.001
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


def train(input_tensor1,input_tensor2,input_tensor3):
    input_tensor1.requires_grad=True
    input_tensor2.requires_grad=True
    input_tensor3.requires_grad=True
    t.requires_grad=True
   # print("input_tensor",input_tensor)
   # hidden = rnn.init_hidden()
    loss=torch.zeros(len(input_tensor1),time_steps)
    output= torch.zeros(len(input_tensor1),time_steps)
    tt=torch.zeros(len(input_tensor1),1)
    #loss=torch.zeros(time_steps,1)
    #output= torch.zeros(time_steps,1)
    
    for i in range(time_steps-3):
        tt[:,:]=t[i+3]
        #inp_tensor=input_tensor.reshape(1,1)
        #print("inp",inp_tensor)
        combined = torch.cat((input_tensor1,input_tensor2,input_tensor3, tt),1)
       # print("i+3",t[i+3])
        #print("combined",combined)
        pred_output = rnn(combined.float())
        y4=pred_output
        #y2=k*delta_t+inp_tensor
        #y2=k
       # print("pred_out",y2)
        #dydt=(y2-input_tensor)/(t[i+1]-t[i])
        dydt=(3*y4-4*input_tensor3+input_tensor2)/(2*delta_t)
        #dydtt=(2*y4-5*input_tensor3+4*input_tensor2-3*input_tensor1)/(delta_t**3)
        #dydtt=()
        #print("dydt",dydt)
        #Psi_t = torch.autograd.grad(y4, combined, grad_outputs=torch.ones_like(y4),
                                     # create_graph=True)[0]
#        print(Psi_t.shape)
        input_tensor1=input_tensor2
        input_tensor2=input_tensor3
        input_tensor3=y4
        loss[:,i]=((dydt-torch.sin(10*t[i+3]))**2).squeeze()
        #loss[:,i]=((Psi_t[:,3]-torch.sin(10*t[i+3]))**2).squeeze()
        #loss[i]=(m*dydtt+c*dydt+k*y4)**2
       # print("loss",loss)
        output[:,i]=pred_output.squeeze()
       # print("i",i,"out",len(output))
    #print("i",i,"output:",j,output.shape)
    #print(torch.sum(loss,1).shape)
    loss1=torch.mean(torch.sum(loss,1))
      
    
    return output, loss1
from sklearn.utils import shuffle



n_iters = 50000
output1=torch.zeros(time_steps,1)
for i in range(n_iters):
    shf_y_train=shuffle(y_train)
    input_signal1,input_signal2,input_signal3=shf_y_train[:,0],shf_y_train[:,1],shf_y_train[:,2]
    input_tensor1=torch.tensor(input_signal1).reshape(len(y_train),1)
    input_tensor2=torch.tensor(input_signal2).reshape(len(y_train),1)
    input_tensor3=torch.tensor(input_signal3).reshape(len(y_train),1)
    #print("epoch",i)
    #first_input= input_tensor[:,0]
    #first_input=first_input.reshape(1,1)
    #print("input_tensor",input_tensor)
    output, loss = train(input_tensor1.float(),input_tensor2.float(),input_tensor3.float())
    output1=torch.cat((input_tensor1,input_tensor2,input_tensor3,output),1)
    output2=output1[:,0:output1.size()[1]-3]
    #print("output",output)
    #print("output1",output1)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if i%100==0:
        print("epochs:",i+1, "loss",loss.item())
        plt.figure(2)      
        plt.plot(t.detach().numpy(),output2[:,:].T.detach().numpy(),"r")
        plt.plot(t.detach().numpy(),shf_y_train.T,"--k")
        #plt.plot(t.detach().numpy(),y,"--k",label="true")
        #plt.plot(t.detach().numpy(),state,"--y",label="velocity state")
        plt.title("solution",)
        plt.legend(loc = 'upper right',facecolor = 'w')
        plt.text(1.065,0.7,"Epoch: %i"%(i+1),fontsize="xx-large",color="k")
  
        plt.show()

with torch.no_grad():
    shf_y_test=shuffle(y_test)
    test_input_signal1,test_input_signal2,test_input_signal3=shf_y_test[:,0],shf_y_test[:,1],shf_y_test[:,2]
    test_input_tensor1=torch.tensor(test_input_signal1).reshape(len(y_test),1)
    test_input_tensor2=torch.tensor(test_input_signal2).reshape(len(y_test),1)
    test_input_tensor3=torch.tensor(test_input_signal3).reshape(len(y_test),1)
    test_output, test_loss = train(test_input_tensor1.float(),test_input_tensor2.float(),test_input_tensor3.float())
    output1=torch.cat((test_input_tensor1,test_input_tensor2,test_input_tensor3,test_output),1)
    output2=output1[:,0:output1.size()[1]-3]
    #print("output",output)
    #print("output1",output1)
    
    plt.figure(4)      
    plt.plot(t.detach().numpy(),output2[:,:].T.detach().numpy(),"r")
    plt.plot(t.detach().numpy(),shf_y_test.T,"--k")
    #plt.plot(t.detach().numpy(),y,"--k",label="true")
    #plt.plot(t.detach().numpy(),state,"--y",label="velocity state")
    plt.title("solution",)
    plt.legend(loc = 'upper right',facecolor = 'w')
    plt.text(1.065,0.7,"Epoch: %i"%(i+1),fontsize="xx-large",color="k")
  
    plt.show()







