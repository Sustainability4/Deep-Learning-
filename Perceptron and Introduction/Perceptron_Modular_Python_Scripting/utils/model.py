import numpy as np
import os
import joblib

class Perceptron:
    # initialising the perceptron class. In order to initialise the perceptron class we will be 
    # providing eta i.e h elearning rate and an epoch that is the total number of cycles one 
    # want forward and backward propogation to run. 

    # Constructor: 
    def __init__(self, eta: float = None, epochs: int = None):
        # Why 3 was taken in np.random.rand()?
        #
        self.weights = np.random.randn(3) * 1e-4 # very very small value
        # Conditional variable is_training : this variable will store the answer to the question
        # whether model is training or not 
        is_training = (eta is not None) and (epochs is not None)
        if is_training:
            print(f"initial weights before training : \n{self.weights}")
        # initialisisng eta and epochs parameters 
        self.eta = eta
        self.epochs = epochs 
    
    
    
    
    # strating a function with an underscore means that the function will be internal and we 
    # will not be utilising it externally. This will return a dot product of inputs and weights
    def _z_outcome(self, inputs, weights):
        return np.dot(inputs,weights) # dot product of the input and weight matrix 
    # This is for the activation function f which will take the value of z obtained from
    # sigma in a forward pass
    def activation_function(self, z):
        return np.where(z>0, 1, 0) # Step function suggesting that if z>0 return 1 else return 0. 

    # to fit in the neural net depending upon the value classification, finiding out the learning
    # parameter or bias 
    def fit(self,X, y):
        # X will be the array of inputs provided to our perceptron
        self.X = X
        self.y = y
        
        # defining the bias in X, bias is created with a column matrix of ones with len(X) number of Rows and 
        # one column
        X_with_bias = np.c_[self.X, np.ones((len(self.X),1))]
        print(f"X with bias : {X_with_bias}")

        for epoch in range(self.epochs):
            print("__"*10)
            print(f"for epoch >> {epoch}")
            print("__"*10)

            # We will calculate the z value first using X_with_bias and weights, actually multiplying them 
            z = self._z_outcome(X_with_bias,self.weights)
            # After we have calculated the z value we will put that in the activation function to calculate y_hat
            # y_hat value will be obtained from the activation function after passing in the value for z 
            y_hat = self.activation_function(z)
            print(f"predicted value after the forward pass : \n{y_hat}")

            #lets calculate the error 
            self.error = self.y-y_hat
            print(f"error is : \n{self.error}")

            # Lets update the weights using weights update rule of perceptron
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            print(f"weight updated after the epoch : {epoch + 1}/{self.epochs} :\n{self.weights}")
            print("##"*10)

    # predicting the values using perceptron based on trained value 
    def predict(self, test_input):
        X_with_bias = np.c_[test_input, np.ones((len(test_input),1))]

        # Calculating the z value using the trained weights above from the perceptron
        z = self._z_outcome(X_with_bias,self.weights)

        return self.activation_function(z)

    # To calulate the losses 
    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"Total Loss : {total_loss}")
        return total_loss

    # internal method as it starts with _ 
    def _create_dir_return_path(self,model_directory, filename):
        os.makedirs(model_directory,exist_ok=True) # If the directory already exists this will not give an error 

        return os.path.join(model_directory,filename)

    # Saving the model we have created 
    def save(self, filename, model_directory = None):
        if model_directory is not None:
            model_file_path = self._create_dir_return_path(model_directory, filename)
            # We will be saving the self which includes weights, eta and epochs to the prescribed model path
            joblib.dump(self,model_file_path)
        else: 
            model_file_path = self._create_dir_return_path("model", filename)
            joblib.dump(self,model_file_path)
            
    def load(self, file_path):
        return joblib.load(file_path)
