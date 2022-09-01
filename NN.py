
from numpy.random import rand, randn
from numpy import tanh, zeros, ones, eye, arctanh
from numpy import concatenate, dot, diag, sum
from numpy.linalg import norm, pinv
from copy import deepcopy


class NeuralNetwork:

    def __init__(self,
                 number_of_inputs,
                 number_of_hidden_neurons,
                 number_of_outputs,
                 activation_function=lambda x: tanh(x),
                 activation_function_derivation=lambda x: 1+tanh(x)**2,
                 output_function=lambda x: x,
                 output_function_derivation=lambda x: ones(x.shape),
                 output_function_inv=lambda x: x,
                 max_epoch = 10000,
                 log=False) -> None:

        # initialize NN values
        self.InputNumber = number_of_inputs
        self.OutputNumber = number_of_outputs
        self.HiddenLayerNumber = len(number_of_hidden_neurons)
        self.HiddenLayerNeurons = number_of_hidden_neurons
        self.ParametersNumber = 0
        self.W = [randn(number_of_hidden_neurons[0], number_of_inputs+1)*0.01];
        self.ParametersNumber += self.W[-1].size
        for i in range(1,self.HiddenLayerNumber):
            self.W.append(randn(number_of_hidden_neurons[i], number_of_hidden_neurons[i-1]+1)*0.1)
            self.ParametersNumber += self.W[-1].size
        self.W.append(randn(number_of_outputs , number_of_hidden_neurons[-1]+1))
        self.ParametersNumber += self.W[-1].size
        self.W_new = deepcopy(self.W)
        self.af = activation_function
        self.daf = activation_function_derivation
        self.of = output_function
        self.dof = output_function_derivation
        self.ofi = output_function_inv
        self.max_epoch = max_epoch
        self.log = log
        pass

    def __feedforward(self, Input):
        # modify input (add a 1 as last input for neurons bias)
        Modified_Input = concatenate((Input, ones((1,Input.shape[1]))))
        
        # all layers inputs are saved here
        # initialize with input 2 layer value
        net = [dot(self.W_new[0] , Modified_Input)]
        
        # all layers outputs are saved here
        # initialize with first layer output
        y = [self.af(net[0])]
        
        # all layers slopes are saved here
        # initialize with first layer slope
        s = [self.daf(net[0])]
        
        for i in range(1 , self.HiddenLayerNumber):
            # modified last calculating layer output to inserting bias effect
            Modified_Input = concatenate((y[-1], ones((1,y[-1].shape[1]))))
            
            # calculate and add last calculating layer input
            net.append(dot(self.W_new[i] , Modified_Input))
            # calculate and add last calculating layer output
            y.append(self.af(net[-1]))
            # calculate and add last calculating layer slope
            s.append(self.daf(net[-1]))
        
        # modified last calculating layer output to inserting bias effect
        Modified_Input = concatenate((y[-1], ones((1,y[-1].shape[1]))))
        # calculate and add last layer input
        net.append(dot(self.W_new[self.HiddenLayerNumber] , Modified_Input))
        # calculate and add last layer output
        y.append(self.of(net[-1]))
        # calculate and add last layer slope
        s.append(self.dof(net[-1]))
        
        return y , net , s

    def __call__(self, Input):
        output, _ , _ = self.__feedforward(Input)
        return output[-1]

    def Train(self, Inputs, Outputs, learning_factor=0.1, thershold=1e-3) -> None:
        # delta list making
        delta = []
        
        # initialize delta
        for i in range(self.HiddenLayerNumber):
            delta.append(zeros((self.HiddenLayerNeurons[i]+1, self.HiddenLayerNeurons[i]+1)))
        delta.append(zeros((self.OutputNumber, self.OutputNumber)))  
        
        
        # Fast Correction
        y , net , s = self.__feedforward(Inputs)
        Modified_Input = concatenate((y[self.HiddenLayerNumber-1], ones((1,y[self.HiddenLayerNumber-1].shape[1]))))
        self.W[self.HiddenLayerNumber] = dot(pinv(Modified_Input.T) , self.ofi(Outputs).T).T
        self.W_new = deepcopy(self.W)
        
        # Forward computation
        y , net , s = self.__feedforward(Inputs)
        errors = y[-1] - Outputs
        max_error = sum(errors**2)**0.5/Inputs.shape[1]
            
        learning_rate_decision = 0
        learning_factor_tmp = learning_factor
        
        if self.log:
            print('{:05} , {:.2E} , {:.20f}'.format(0, learning_factor, max_error))
         
        epoch = 0   
        while  max_error > thershold and epoch < self.max_epoch:
            epoch += 1
            delta_wp = zeros((self.ParametersNumber,1))  
            H = eye(self.ParametersNumber)*learning_factor_tmp
            
            # Back Propagation for Jacobian
            for p in range(Inputs.shape[1]):
                J = zeros((self.OutputNumber , self.ParametersNumber))

                param_index = 0;
                delta[self.HiddenLayerNumber] = diag(s[self.HiddenLayerNumber][:,p])
                for j in range(self.W[self.HiddenLayerNumber].shape[0]):
                    for k in range(self.W[self.HiddenLayerNumber].shape[1]-1):
                        J[:,param_index] = delta[self.HiddenLayerNumber][:,j]*y[self.HiddenLayerNumber-1][k,p]
                        param_index+=1
                    J[:,param_index] = delta[self.HiddenLayerNumber][:,j]
                    param_index+=1
                
                for i in range(self.HiddenLayerNumber-1,-1,-1):
                    delta[i] = dot(dot(delta[i+1], self.W[i+1][:,:-1]), diag(s[i][:,p]))
                    for j in range(self.W[i].shape[0]):
                        for k in range(self.W[i].shape[1]-1):
                            if i == 0:
                                J[:,param_index] = delta[i][:,j]*Inputs[k,p]
                            else:
                                J[:,param_index] = delta[i][:,j]*y[i-1][k,p]
                            param_index+=1
                        J[:,param_index] = delta[i][:,j]
                        param_index+=1
                H += dot(J.T , J)
                delta_wp += dot(J.T , errors[:,p].reshape((-1,1)))

            # Levenberge-Marquart Algorithm
            delta_w = dot(pinv(H) , delta_wp)
            
            # Gradient Descent (LMA Like)
            #delta_w = (delta_wp /norm(H))/ learning_factor_tmp
            
            # update parameters    
            param_index = 0;
            for i in range(self.HiddenLayerNumber,-1,-1):
                for j in range(self.W[i].shape[0]):
                    for k in range(self.W[i].shape[1]):
                        self.W_new[i][j,k] -= delta_w[param_index , 0]
                        param_index+=1
                        
            # Forward computation           
            new_y , new_net , new_s = self.__feedforward(Inputs)
            new_errors = new_y[-1] - Outputs
            new_max_error = sum(errors**2)**0.5/Inputs.shape[1]
            
            
            if (new_max_error > max_error):
                learning_rate_decision += 1
                if learning_rate_decision > 10:
                    learning_rate_decision = 0
                    learning_factor_tmp *= 10
                
                    
            self.W = deepcopy(self.W_new)
            y = new_y
            net = new_net
            s = new_s
            errors = new_errors
            max_error=new_max_error
            
            if self.log:
                print('{:05} , {:.2E} , {:.20f}'.format(epoch, learning_factor_tmp, new_max_error))

        pass

if __name__ == '__main__':
    from math import pi
    import numpy
    from matplotlib import pyplot as plt
    
    N = 100;
    Inputs_Delay = 4;
    func = lambda x: numpy.sin(x*pi/180)
    time = numpy.arange(N).reshape((1,-1))*9
    time_delayed = numpy.array([numpy.arange(N)-1-i for i in range(Inputs_Delay)])*9
    Outputs = func(time)
    Inputs = func(time_delayed)
    #print(Inputs)
    #print(Outputs)
    NN = NeuralNetwork(
              number_of_inputs=Inputs_Delay,
              number_of_hidden_neurons=[2],
              number_of_outputs=1, 
              #activation_function=lambda x: 2/(1+numpy.exp(-2*x))-1,
              #activation_function_derivation=lambda x: 1+(2/(1+numpy.exp(-2*x))-1)**2,
              max_epoch=1000,
              log=True)
    
    fig1 = plt.figure("Figure 1")
    plt.plot(time[0,:] , Outputs[0,:] , time[0,:] , NN(Inputs)[0,:])
    
    NN.Train(Inputs=Inputs, Outputs=Outputs, learning_factor=1e+10, thershold=1e-10)

    fig2 = plt.figure("Figure 2")
    plt.plot(time[0,:] , Outputs[0,:] , time[0,:] , NN(Inputs)[0,:])

    plt.show()