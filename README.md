# NeuralNetworkFromScratch
A simple python class for neural network from scratch

Hi everyone

It's a simple python code which contains a MLP neural network and wrote from scratch. It's training algorithm is Marquart-Levenbeurg method.

## How to use

#### 1. Download and copy beside your main python code

#### 2. import it like below:
```
from NN import NeuralNetwork
```

#### 3. Make a NN in your code
```
myNN = NeuralNetwork(
              number_of_inputs= place number of inputs here,
              number_of_hidden_neurons= [h1,h2,....], # place each layer number of neurons as hi e.g [5,5]
              number_of_outputs= place number of inputs here, 
              #activation_function=lambda x: 2/(1+numpy.exp(-2*x))-1, # you can set any activation function you like
              #activation_function_derivation=lambda x: 1+(2/(1+numpy.exp(-2*x))-1)**2, # wrote down activation function derivation
              max_epoch=100, # number of epochs, default is 1000
              #log=True, #if you want see some log in training
              )
```

#### 4. Train NN 
"Inputs" are inputs data for neural network and "Outputs" are target neural network must reach as it output for each input
```
NN.Train(Inputs=Inputs, Outputs=Outputs, learning_factor=1e+10, thershold=1e-10)
```

### There is an example in main part of code if you need

Good luck!
