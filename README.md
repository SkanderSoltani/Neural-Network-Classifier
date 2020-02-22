## Welcome to my Github page: 
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

<img src="https://latex.codecogs.com/svg.latex?\Large&space;for l in layers:" title="for l in layers:" />
<img src="https://latex.codecogs.com/svg.latex?\Large&space;Z_{1}=W_{1}'X" title="\Large Z_{1}=W_{1}'X" />
<img src="https://latex.codecogs.com/svg.latex?\Large&space;A_{1}=g(Z_{1})" title="\Large A_{1}=g(Z_{1})" />
<img src="https://latex.codecogs.com/svg.latex?\Large&space;Z_{2}=W_{2}'A_{1}" title="\Large Z_{2}=W_{2}'A_{1}" />
<img src="https://latex.codecogs.com/svg.latex?\Large&space;A_{2}=AL=sigmoid(Z_[2})" title="\Large A_{2}=AL=sigmoid(Z_{2})" />

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

