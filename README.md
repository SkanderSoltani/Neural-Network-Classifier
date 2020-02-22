## Ride Hard or Stay Home -- Welcome to my Github Page 
![](Memory.jpg)


### Project Description
LogisticAnn is a personal project/memo developed with the goal to thoroughly study Neural Networks along with Backpropagation and a plethora of gradient based optimization techniques used in the field.  During this journey, we will also derive all the formulas and implement the ensemble in a python class called LogisticAnn. The longer goal is to learn the mathematics underlying Neural Networks enough to be able to design and implement bespoke solutions adapted to specific problems as needed. In addition, this memo will serve as a reference point for upcoming projects. 

### Definition of Neural Networks
Artificial neural networks or connectionist systems are computing systems inspired by the biological neural networks that constitute animal brains. Such systems "learn" to perform tasks by considering examples, generally without being programmed with task-specific rules.

### Components:
![](NN.png)

Neural Networks consist of 3 type of layers: 1. Input layer, 2. Hidden layers, 3. Output layer. In the example above it is a one-layer NN (We do not count input & output layers). Each layer consists of one or several neurons (blue circles in the diagram above) that get activated by an activation function. There are several activation functions used, in this project, our class gives the option to either use “tanh – Hyperbolic Tangent” or ” Relu – Rectified Linear Unit” functions for the hidden layers and will use the sigmoid for the output layer. 

### Computation:
There are three steps in the computation process of Neural Networks: 
The optimization underlying Neural Networks computations is based on gradients. Hence, there are 2 main steps in the optimization: 
#### 1. Initialization:
Initialization of the weights is done once; only at the beggining of the process. The initialization process can have drastic impact on the convergence as well as the speed of the algorithm. We chose to initialize our weights such that are normally distributed with mean of 0 and variance.

#### 2.The Forward Propagation:  
  The forward propagation algorithms’ task is to propagate the information forward from the input to the output layers using a          combination of two functions of which one is linear, the second is the activation function. The algorithm describing the forward propagation process for a one hidden layer network is as follow:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Z_{1}=W_{1}'X+b_{1}" title="\Large Z_{1}=W_{1}'X + b_[1}" />

<img src="https://latex.codecogs.com/svg.latex?\Large&space;A_{1}=g(Z_{1})" title="\Large A_{1}=g(Z_{1})" />

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Z_{2}=W_{2}'A_{1}+b_{2}" title="\Large Z_{2}=W_{2}'A_{1}+b_{2}" />

<img src="https://latex.codecogs.com/svg.latex?\Large&space;A_{2}=AL=sigmoid(Z_{2})" title="\Large A_{2}=AL=sigmoid(Z_{2})" />

Where:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;X:}" title="\Large X:" /> <- is the input matrix with dimensions (n,m); "n" being the number of features and "m" the training sample size.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;b_{i}:}" title="\Large b_{i}:" /> <- is the bias unit at each layer of size <img src="https://latex.codecogs.com/svg.latex?\Large&space;(n_{l},m)}" title="\Large (n_{l},m)" /> where <img src="https://latex.codecogs.com/svg.latex?\Large&space;'n_{l}'}" title="\Large (n_{l},m)" /> is size of layer <img src="https://latex.codecogs.com/svg.latex?\Large&space;'l'}" title="\Large 'l'" /> and "m" is the training sample size. 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;W_{l}:}" title="\Large W_{l}:" /> <- is the weight matrix of size <img src="https://latex.codecogs.com/svg.latex?\Large&space;(n_{l-1},n_{l})}" title="\Large (n_{l-1},n_{l})" /> where <img src="https://latex.codecogs.com/svg.latex?\Large&space;n_{l-1}}" title="\Large n_{l-1}" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;n_{l}}" title="\Large n_{l}" /> represent the size of the previous and current layer respectively. <img src="https://latex.codecogs.com/svg.latex?\Large&space;W_{l}}" title="\Large W_{l}" /> are also the target variables of our problem.  

<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x):}" title="\Large g(x):" /> <-  is the activation function for layer <img src="https://latex.codecogs.com/svg.latex?\Large&space;l}" title="\Large l" />. Our class gives the option to use a "relu" or "tanh" functions.<br>
<u>relu:</u> <br><img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x)=(Z_{l},0)^{+}" title="\Large g(x)=(Z_{l},0)^{+}" /><br>

<u>tanh:</u>,br> <img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x)=\frac{sinh(x)}{cosh(x)}" title="\Large g(x)=\frac{sinh(x)}{cosh(x)}" />

The method responsible to propagate information from layer to layer is described below: 
```python
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

```
#### 2.The Forward Propagation: 
\begin{equation}
\sqrt{2}
$\sqrt{2}$.

\end{equation}



```markdown
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
\sqrt{2}
$\sqrt{2}$.
# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).



### Support or Contact

