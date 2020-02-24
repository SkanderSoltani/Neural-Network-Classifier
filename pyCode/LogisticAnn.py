import numpy as np
#import plotly.express as px
import pandas as pd
import math
from numpy import linalg as LA

class LogisticAnn:

    
    
    def __init__(self,layerDim):
        self.type       = "Logistic"
        self.layerDim   = layerDim
        self.parameters = None
        self.gradients  = None
        self.activation = None
        
    ####################
    # Helper Functions #
    ####################

    def _initialize_parameters(self,layer_dim):
        """
        Arguments:
            layer_dim  <- a list containing the dimentions of each layer in the network
        Returns:
            Parameters <- a dictionary containing parameters "W1","b1".."Wl","bl"...."WL","bL". where "L" is the number of the final output layer in the network
                            "Wl" - Matrix of weights of shape (layer_dim(l),layer_dim(l-1))
                            "bl  - Vector of the bias nodes of shape (leyer_dim(l),1)
        """
        np.random.seed(3) # Fixing the seed in order to get the same result (helps for debugging)
        parameters = {}
        L = len(layer_dim) 
        for l in range(1,L):
            parameters["W" + str(l)] = np.random.randn(layer_dim[l],layer_dim[l-1]) *  2 / np.sqrt(layer_dim[l-1])# the purpose of the perturb is to make the initialized weights very close to 0
            parameters["b" + str(l)] = np.zeros((layer_dim[l],1))
            # Just to make sure that each weight and bias matrix is of the appropriate shape
            assert(parameters["W" + str(l)].shape == (layer_dim[l],layer_dim[l-1]))
            assert(parameters["b" + str(l)].shape == (layer_dim[l],1))
        
        return parameters

    def _relu(self,Z):
        return(np.maximum(0,Z))

    def _sigmoid(self,Z):
        return (1/(1+np.exp(-Z)))

    # tanh is available in numpy library and can be called 
    
    def _relu_backward(self,dA,cache):
        """
        Implements backpropagation for a single "relu" unit. It returns dZ_prev = dA_curr * g'(Z_prev) where function g() is a relu function

        Arguments:
            dA    <- post activation of current layer
            cache <- cache tuple of the current unit generated from L_forwardProp containing (A_prev,W,b,Z)
        Results:
            dZ    <- dL/dZ gradient of the cost with respect to Z of the current layer

        """
        Z = cache[3]
        dZ = np.array(dA,copy=True)

        dZ[Z<0] = 0

        assert(dZ.shape==Z.shape) # Just to make sure that the partial derivative matrix dZ shape is similar to Z

        return dZ

    def _sigmoid_backward(self,dA,cache):
        """
        Implements backpropagation for a single "sigmoid" unit. It returns dZ_prev = dA_curr * g'(Z_prev) where function g() is a sigmoid function

        Arguments:
            dA    <- post activation of current layer
            cache <- cache tuple of the current unit generated from L_forwardProp containing (A_prev,W,b,Z)
        Results:
            dZ    <- dL/dZ gradient of the cost with respect to Z of the current layer

        """
        Z = cache[3]

        s = self._sigmoid(Z)

        dZ = dA * s * (1-s)

        assert(dZ.shape == Z.shape)

        return dZ 

    def _tanh_backward(self,dA,cache):
        """
        Implements backpropagation for a single "tanh" unit. It returns dZ_prev = dA_curr * g'(Z_prev) where function g() is a tanh function

        Arguments:
            dA    <- post activation of current layer
            cache <- cache tuple of the current unit generated from L_forwardProp containing (A_prev,W,b,Z)
        Results:
            dZ    <- dL/dZ gradient of the cost with respect to Z of the current layer

        """
        Z = cache[3]

        # tanh'(Z) = 1-tanh(Z)^2  proof is simple sinh'(Z)=cosh(Z), cosh'(Z)=sinh(Z)   tanh = sinh/cosh  => tanh' = (sinh' * cosh - cosh' * sinh) / cosh^2 =>  tanh' = (cosh^2 - sinh^2)/cosh^2 = 1-tanh^2
        dZ = dA * (1 - (np.tanh(Z)**2) )

        assert(dZ.shape == Z.shape)

        return dZ

    def _forwardProp(self,A_prev, W, b, activation):
        """
        Implementation of the forward propagation algorithm
        Arguments:
            A_prev     <- The activation matrix results of the previous layer of shape (number of nodes in the previous layer, m)
            W          <- Matrix of weights of the current layer of shape (number of nodes in the current layer, number of nodes in the previous layer)
            b          <- Matrix of bais units of the current layer of shape (number of nodes in the current layer,1)
            activation <- The activation function used to activate the current layer. stored as a text string: "sigmoid","relu" or "tanh"
        Returns:
            A          <- The activation matrix results of the current layer of shape (number of nodes in the current layer, m)
            cache      <- a tuple containing (A_prev,W,b,Z)
        """
        Z = np.dot(W,A_prev) + b # the linear activation
        if activation == "sigmoid":
            A = self._sigmoid(Z)
        elif activation == "relu":
            A = self._relu(Z)
        elif activation == "tanh":
            A = np.tanh(Z)
        
        cache = (A_prev,W,b,Z)
        
        return A,cache

    def _L_model_forwardProp(self,X,parameters,activation):
        """
        Implements forward propagation acctoss all the layers of the network - Linear-> activation (tanh or relu) (L-1)  -> sigmoid (L)
        Arguments:
            X <- input data, np array of shape (number of parameters, number of examples)
            activation <- activation function of hidden layers. stores string : "relu" or "tanh"
        Returns:
            AL <- activation units of the last layer shape (number of classes | 1 if binary classification,sample size)
            caches <- list of caches of all forward prop layers
        """
        caches=[]
        A_prev = X
        L = len(parameters) // 2 # number of layers in the network
        for l in range(1,L):
            W = parameters["W"+str(l)]
            b = parameters["b"+str(l)]
            A, cache = self._forwardProp(A_prev, W, b, activation)
            A_prev = A
            caches.append(cache)
        
        # implementing sigmoid function for the last layer L
        W = parameters["W"+str(L)]
        b = parameters["b"+str(L)]
        AL,cache = self._forwardProp(A_prev, W, b, activation="sigmoid")
        caches.append(cache) # Appending cache from last list

        return AL,caches
    
    def _linear_backward(self,dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b,Z = cache
        m = A_prev.shape[1]

        
        dW = 1/m * np.dot(dZ,A_prev.T)
        db = 1/m * np.sum(dZ,axis=1,keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dW,db,dA_prev 


    def _backProp(self,dA,cache,activation):
        """
        Implement the backward propagation using linear backprop function.
        
        Arguments:
        dA         <- post-activation gradient for current layer l 
        cache      <- a tuple containing (A_prev,W,b,Z) of the current layer
        activation <- the activation to be used in this layer, stored as a text string: "sigmoid", "relu" or "tanh"
        
        Returns:
        dA_prev    <- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW         <-Gradient of the cost with respect to W (current layer l), same shape as W
        db         <-Gradient of the cost with respect to b (current layer l), same shape as b
        """
        
        
        if activation == "sigmoid":
            dZ = self._sigmoid_backward(dA,cache)
            dW,db,dA_prev = self._linear_backward(dZ,cache)
            return dW,db,dA_prev
        elif activation == "relu":
            dZ = self._relu_backward(dA,cache)
            dW,db,dA_prev = self._linear_backward(dZ,cache)
            return dW,db,dA_prev
        elif activation == "tanh":
            dZ = self._tanh_backward(dA,cache)
            dW,db,dA_prev = self._linear_backward(dZ,cache)
            return dW,db,dA_prev

    def _L_model_backProp(self,AL, Y ,activation ,caches,lambd):
        """
        Implement the backward propagation for the (relu or tanh) * (L-1) -> LINEAR -> SIGMOID last layer L
        
        Arguments:
        AL         <- probability vector, output of the forward propagation (L_model_forwardProp())
        Y          <- true "label" target variable in the supervised learning problem size (1,m)
        Activation <- the activation to be used in the hidden, stored as a text string:  "relu" or "tanh"
        caches     <- list of caches containing all caches for each layer. each cache is a tuple containing  (A_prev,W,b,Z) 

                    
        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ... 
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ... 
        """
        grads ={}
        L = len(caches) # The number of layers
        m = AL.shape[1] # number of training examples
        Y = Y.reshape(AL.shape)

        epsilon = 1e-7

        # Starting back propagation by computing dAL given a sigmoid activation function for the last layer L  
        dAL = np.divide(-Y,AL+epsilon) + np.divide((1-Y),(1-AL)+epsilon)
        
        # retrieving the cache tuple of layer L
        cache_L = caches[L-1]
        _, W, _,_ = cache_L
        # getting the gradient of layer L
        grads["dW"+str(L)],grads["db"+str(L)],grads["dA"+str(L-1)] = self._backProp(dA=dAL,cache = cache_L,activation="sigmoid")
        grads["dW"+str(L)] = grads["dW"+str(L)] + (lambd/m) * W

        #loop from l=L-2 to l=0 to calculate the rest of the gradient through each layer
        for l in reversed(range(L-1)):
            cache_l = caches[l]
            _, W, _,_ = cache_l
            dW_temp,db_temp,dA_prev_temp =  self._backProp(grads["dA"+str(l+1)],cache_l,activation)
            grads["dA"+str(l)]   = dA_prev_temp
            grads["dW"+str(l+1)] = dW_temp + (lambd/m) * W
            grads["db"+str(l+1)] = db_temp

        return grads
    
    def _compute_cost(self,AL , Y, parameters,lambd):
        """
        Implement the cost function defined by -1/m sum(ylog(a)+(1-y)log(1-a))


        Arguments:
        AL         <- probability vector corresponding to your label predictions, shape (n, number of examples) where n=1 if problem binary n=number of labels if problem has several labels
        Y          <- target variable in the supervising problem, shape (n, number of examples) where n=1 if problem binary n=number of labels if problem has several labels
        parameters <- a python dictionary of parameters W1, b1,...,Wn,bn
        lambd      <- penalty term
        Returns:
        J  <- cross-entropy cost
        """

        epsilon = 1e-7
        m = AL.shape[1]
        L = len(parameters) // 2
        penalty_term = np.sum([LA.norm(parameters["W"+str(l+1)],"fro")**2 for l in range(L)])
        
        J = (-1/m) * (Y * np.log(AL+epsilon) + (1-Y)* (1) * (np.log(1-AL+epsilon))).sum() + lambd/(2*m) * penalty_term

        J = np.squeeze(J) 
        assert(J.shape == ())
        return J

    def _update_parameters(self,parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        Parameters <- a dictionary containing parameters "W1","b1".."Wl","bl"...."WL","bL". where "L" is the number of the final output layer in the network
                            "Wl" - Matrix of weights of shape (layer_dim(l),layer_dim(l-1))
                            "bl  - Vector of the bias nodes of shape (leyer_dim(l),1)
        grads -- python dictionary containing your gradients, output of L_model_backward
        
        Returns:
        parameters -- Updated parameters
        """
        
        L = len(parameters) // 2 

        # Update rule for each parameter. Use a for loop.
        
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]   # updating the weight matrices
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]   # updating the bias units
    
        return parameters

    def _dictionary_to_vector(self,parameters,grad=False):
        """
        Roll all our parameters dictionary into a single vector satisfying our specific required shape.
        """
        keys = []
        count = 0
        param = {}
        
        if grad == True:
            L = len(parameters) // 3
            for l in range(L):
                param["dW" + str(l+1)] = parameters["dW" + str(l+1)]
                param["db" + str(l+1)] = parameters["db" + str(l+1)]
        else:
            param = parameters

        for key in param.keys():
            # flatten parameter
            new_vector = np.reshape(param[key], (-1,1)) # Unroll the gardient vector
            keys = keys + [key] * new_vector.shape[0] # Clist concatenation
            
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1
        
        return theta, keys

    def _vector_to_dictionary(self,theta,layer_dim,grad=False):
        """
        Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
        Arguments:
            theta      <- vector of floats representing the gradients
            layer_dim  <- a list containing the dimentions of each layer in the network
        Result:
            parameters <- a python dictionary representing the pareameters "W1","b1","W2" ....
        """
        
        L = len(layer_dim) - 1 #number of layers
        index=0
        parameters = {}
        if grad==False:
            for l in range(L):
                parameters["W" + str(l+1)] = theta[index:(index + layer_dim[l+1]*layer_dim[l])].reshape((layer_dim[l+1],layer_dim[l]))
                index = index + layer_dim[l+1]*layer_dim[l]
                parameters["b" + str(l+1)] = theta[index:(index + layer_dim[l+1])].reshape((layer_dim[l+1],1))
                index = index +  layer_dim[l+1]
        else:
            for l in range(L):
                parameters["dW" + str(l+1)] = theta[index:(index + layer_dim[l+1]*layer_dim[l])].reshape((layer_dim[l+1],layer_dim[l]))
                index = index + layer_dim[l+1]*layer_dim[l]
                parameters["db" + str(l+1)] = theta[index:(index + layer_dim[l+1])].reshape((layer_dim[l+1],1))
                index = index +  layer_dim[l+1]

        return parameters

    def _convertYtoMx(self,Y):
        """
        
        Converts Y variable to np.array (l,m) where l is the number of labels & m is the sample size
        
        """
        labels = np.unique(Y)
        m = Y.shape[1]
        newY = np.zeros((labels.shape[0],m))
        for i in range(m) :
            newY[Y.iloc[0,i],i] = 1
        
        return(newY)

    #/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
    ########### OPTIMIZATION #################
    # MINI BATCH TECHNIQUE
    def _random_mini_batches(self,X, Y, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """
        
        np.random.seed(seed)            # To make your "random" minibatches the same as ours
        m = X.shape[1]                  # number of training examples
        mini_batches = []
        l = Y.shape[0]    
        # Step 1: Shuffle (X, Y) # to make sure that they are all from the same distribution
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((l,m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            
            mini_batch_X = shuffled_X[:,(k)*(mini_batch_size):(k+1)*(mini_batch_size)]
            mini_batch_Y = shuffled_Y[:,(k)*(mini_batch_size):(k+1)*(mini_batch_size)]
            
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            
            mini_batch_X = shuffled_X[:,(k)*(mini_batch_size):shuffled_X.shape[1]]
            mini_batch_Y = shuffled_Y[:,(k)*(mini_batch_size):shuffled_X.shape[1]]
            mini_batch   = (mini_batch_X, mini_batch_Y)
            # append mini batches
            mini_batches.append(mini_batch)
        
        return mini_batches

    ## GRADIENT DESCENT WITH MOMENTUM
    def _initialize_velocity(self,parameters):
        """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        
        Returns:
        v -- python dictionary containing the current velocity.
                        v['dW' + str(l)] = velocity of dWl
                        v['db' + str(l)] = velocity of dbl
        """
        
        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}
        
        # Initialize velocity
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
            v["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
            
        return v
    # update parameters using gradient descent with momentum
    def _update_parameters_with_momentum(self,parameters, grads, v, beta, learning_rate):
        """
        Update parameters using Momentum
        
        Arguments:
        parameters    <- python dictionary containing your parameters
        grads         <- python dictionary containing your gradients for each parameters
        v             <- python dictionary containing the current velocity
        beta          <- the momentum hyperparameter, scalar
        learning_rate <- the learning rate, scalar
        
        Returns:
        parameters    <- python dictionary containing your updated parameters 
        v             <- python dictionary containing your updated velocities
        """

        L = len(parameters) // 2 # number of layers in the neural networks
        
        # Momentum update for each parameter
        for l in range(L):
            # compute velocities (Exponential weighting)
            v["dW" + str(l+1)] = beta * (v["dW" + str(l+1)]) + (1-beta) * grads["dW" + str(l+1)]
            v["db" + str(l+1)] = beta * (v["db" + str(l+1)]) + (1-beta) * grads["db" + str(l+1)]
            # update parameters
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
            
        return parameters, v

    ## Adam Optimizer
    #initialize_adam
    def _initialize_adam(self,parameters) :
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Arguments:
        parameters <- python dictionary containing your parameters.
        
        Returns: 
                v <- python dictionary that will contain the exponentially weighted average of the gradient.
                    
                    
                s <- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    

        """
        
        L = len(parameters) // 2 # number of layers in the neural networks
        v = {}
        s = {}
        
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
            v["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
            s["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
            s["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
        
        
        return v, s

    # pdate_parameters_with_adam

    def _update_parameters_with_adam(self,parameters, grads, v, s, t, learning_rate,
                                    beta1, beta2,  epsilon):
        """
        Update parameters using Adam
        
        Arguments:
        parameters    <- python dictionary containing your parameters:             
        grads         <- python dictionary containing your gradients for each parameters             
        v             <- Adam variable, moving average of the first gradient, python dictionary
        s             <- Adam variable, moving average of the squared gradient, python dictionary
        t             <- counts the number of steps taken by adam 
        learning_rate <- the learning rate, scalar.
        beta1         <- Exponential decay hyperparameter for the first moment estimates 
        beta2         <- Exponential decay hyperparameter for the second moment estimates 
        epsilon       <- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters    <- python dictionary containing your updated parameters 
        v             <- Adam variable, moving average of the first gradient, python dictionary
        s             <- Adam variable, moving average of the squared gradient, python dictionary
        
        """
        
        L = len(parameters) // 2                 # number of layers in the neural networks
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary
        
        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            v["dW" + str(l+1)] = (beta1 * (v["dW" + str(l+1)]) + (1-beta1) * grads["dW" + str(l+1)]) 
            v["db" + str(l+1)] = (beta1 * (v["db" + str(l+1)]) + (1-beta1) * grads["db" + str(l+1)]) 
    

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l+1)] = v["dW"+str(l+1)] / (1-beta1**t)
            v_corrected["db" + str(l+1)] = v["db"+str(l+1)] / (1-beta1**t)
            

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            s["dW" + str(l+1)] = (beta2 * (s["dW" + str(l+1)]) + (1-beta2) * grads["dW" + str(l+1)]**2)
            s["db" + str(l+1)] = (beta2 * (s["db" + str(l+1)]) + (1-beta2) * grads["db" + str(l+1)]**2)
        

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-beta2**2)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-beta2**2)
            

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)]) / (np.sqrt(s_corrected["dW" + str(l+1)])+epsilon)
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)]) / (np.sqrt(s_corrected["db" + str(l+1)])+epsilon)
            

        return parameters, v, s

    


    def fit(self,X, Y,optimizer,layers_dims=None,learning_rate=0.0007, num_epochs=10000, mini_batch_size = 64, beta1=0.9, beta2=0.999, epsilon=1e-8, print_cost=False, activation="relu",lambd = 0): 
        """
        Implements a L-layer neural network: with "tanh" or "relu" activation functions for the hidden layers. "sigmoid" function is used for the activation of the final layer 
        
        Arguments:
        X              <- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y              <- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims    <- list containing the input size and each layer size, of length (number of layers + 1).
        optimizer      <- optimizer takes the following strings "gd", "momentum","adam"
        learning_rate  <- learning rate of the gradient descent update rule
        num_epochs     <- number of epochs
        mini_batch_size<- mini batch size
        beta1          <- exp decay hyperparameter for the first moment estimate 
        beta2          <- exp decay hyperparameter for the second moment estimate
        epsilon        <- hyper parameter preventing division over zero in Adam optimizer       
        print_cost     <- if True, it prints the cost every 100 steps
        
        Returns:
        parameters     <- parameters learnt by the model. They can then be used to predict.
        
        """
        self.activation = activation
        if layers_dims == None:
            layers_dims = self.layerDim
        
        np.random.seed(1)
        costs = []                         # keep track of cost
        t = 0 
        seed = 5
        m = Y.shape[1]
        
        #initializing parameters
        parameters = self._initialize_parameters(layers_dims)

        # Initialize the parameters for the optimizers
        if optimizer =="gd":
            pass
        elif optimizer == "momentum":
            v = self._initialize_velocity(parameters)
        elif optimizer == "adam":
            v, s = self._initialize_adam(parameters)
    
        
        # Loop for optimization
        for i in range(0,num_epochs):
            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = self._random_mini_batches(X=X, Y=Y, mini_batch_size=mini_batch_size, seed=seed)
            cost_total = 0
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                AL, caches = self._L_model_forwardProp(X=minibatch_X, parameters=parameters, activation=activation)
            
                # Compute cost.
                cost_total += self._compute_cost(AL, minibatch_Y,parameters=parameters,lambd=lambd)
            
                # Backward propagation.
                grads = self._L_model_backProp(AL, minibatch_Y,activation ,caches,lambd)
            
                # Update parameters
                if optimizer=="gd":
                    parameters = self._update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)
                elif optimizer=="momentum":
                    parameters, v = self._update_parameters_with_momentum(parameters=parameters,grads=grads,v=v,beta=beta1,learning_rate=learning_rate)
                elif optimizer=="adam":
                    t+=1 # Adam counter for correction
                    parameters, v, s = self._update_parameters_with_adam(parameters=parameters,grads=grads,v=v,s=s,t=t,learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon)
            
            avg_cost = cost_total / m   
            # updating parameters and gradients
            self.gradients  = grads
            self.parameters = parameters

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after epoch %i: %f" %(i, avg_cost))
            if print_cost and i % 100 == 0:
                costs.append(avg_cost)
                
        # plot the cost
        df = pd.DataFrame(costs)
        df["num epochs"] = range(0,num_epochs,100) 
        df.columns = ["Cost","Num Epochs"]
        # fig = px.line(df, x="num_iteration", y="Cost",title='Cost Function vs. Num of Iter')
        # fig.show()
        
        return parameters,df

    def gradient_check_n(self, X, Y, activation ,epsilon = 1e-7,lambd = 0):
        """
        Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
        
        Arguments:
        parameters <- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
        gradients  <- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
        x          <- input datapoint, of shape (input size, 1)
        y          <- true "label"
        epsilon    <- tiny shift to the input to compute approximated gradient with formula(1)
        
        Returns:
        difference <- difference (2) between the approximated gradient and the backward propagation gradient
        """
        parameters = self._initialize_parameters(self.layerDim)
        AL,caches = self._L_model_forwardProp(X=X,activation=activation,parameters=parameters)
        gradients = self._L_model_backProp(activation=activation,AL=AL,Y=Y,caches=caches,lambd=lambd)

        # Set-up variables
        parameters_values, keys = self._dictionary_to_vector(parameters)
        grad,_= self._dictionary_to_vector(gradients,grad=True)
        num_parameters = parameters_values.shape[0]
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))
        
        # Compute gradapprox
        for i in range(num_parameters):
            # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
            # "_" is used because the function you have to outputs two parameters but we only care about the first one
            thetaplus = np.copy(parameters_values)                                      
            thetaplus[i][0] = thetaplus[i][0] + epsilon                                
            AL,_ = self._L_model_forwardProp(X,self._vector_to_dictionary(thetaplus,self.layerDim),activation) 
            J_plus[i] = self._compute_cost(AL=AL,Y=Y,parameters = self._vector_to_dictionary(thetaplus,self.layerDim),lambd=lambd)                               
            
            
            # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
            thetaminus = np.copy(parameters_values)                                     # Step 1
            thetaminus[i][0] = thetaminus[i][0] - epsilon                               # Step 2        
            AL, _ = self._L_model_forwardProp(X,self._vector_to_dictionary(thetaminus,self.layerDim),activation)  
            J_minus[i] = self._compute_cost(AL=AL,Y=Y,parameters = self._vector_to_dictionary(thetaminus,self.layerDim),lambd=lambd)                           
        
            # Compute gradapprox[i]
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2*epsilon)
        
        
        # Compare gradapprox to backward propagation gradients by computing difference.
        numerator =   np.linalg.norm((grad-gradapprox),ord=None)                                           
        denominator = np.linalg.norm((gradapprox),ord=None) + np.linalg.norm((grad),ord=None)                                         
        difference = numerator / denominator                                          
        

        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works! difference = " + str(difference) + "\033[0m")
        
        return difference

    def predict(self,X):
        parameters = self.parameters
        activation = self.activation
        AL,_ = self._L_model_forwardProp(X,parameters,activation)
        if (AL.shape[0]==1):
            return AL
        else:
            probs = pd.DataFrame(AL)
            Y_hat = probs.idxmax(axis=0).values
            return Y_hat , AL


    