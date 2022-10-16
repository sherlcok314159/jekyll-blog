---
title: Introduction to Back-propagation 
date: 2022-09-07 11:10:00 +0800
categories: [Training, Deep Learning]
tags: [Gradients, Optimization]
math: true
mermaid: false
image:
  path: /2022/09/07/uajAxEbG9kCe7hM.jpg 
  width: 600
  height: 100
---
### 1. Gradient descent optimization

Gradient-based methods make use of the gradient information to adjust the parameters. Among them, gradient descent can be the simplest. Gradient descent makes the parameters to walk a small step in the direction of the negative gradient.

$$
\boldsymbol{w}^{\tau + 1} = \boldsymbol{w}^{\tau} - \eta \nabla_{\boldsymbol{w}^{\tau}} E \tag{1.1}
$$

where $\eta, \tau, E$ label learning rate ($\eta > 0$), the iteration step and the loss function. Wait! But why is the negative gradient? Recall the calculus basics

$$
\begin{align}
\nabla_{\boldsymbol{w^{\tau}}}E > 0 \implies E \uparrow \tag {1.2}\\
\nabla_{\boldsymbol{w^{\tau}}}E < 0 \implies E \downarrow \tag{1.3}
\end{align}
$$

Take $\displaystyle y=\frac{1}{2}x^2$ as an example, when $f'(x) < 0$, we can find the function is decreasing. When added by the gradient, $x$ should decrease which is opposite to our desire. Thus, we want to add the negative gradient to $x$. 

![backprop0.png](2022/09/07/3lhCmiytvPuWnRk.png){: w="600" h="900" }

But how to compute the gradients needs a powerful technique: back-propagation.

### 2. Definition of back-propagation

> Back-propagation allows information from the cost to then flow backwards through the network, in order to compute the gradients used to adjust the parameters.
{: .prompt-tip }

Back-propagation can be new to the novices, but it does exist in the life widely. For instance, the loss can be your teacher's attitude towards you. If you fail in one examination, your teacher can be disappointed with you. Then, he can tell your parents about your failure. Your parents then ask you to work harder to win the examination.

Your parents can be seen as **hidden units** in the neural network, and you are the parameter of the network. Your teacher's bad attitude towards your failure can ask you to make adjustments: working harder. Similarly, the loss can require the parameters to make adjustments via gradients.

### 3. Chain Rule

Suppose $z = f(y), y = g(x) \implies z = (f \circ g)(x)$, how to calculate the derivative of $z$ with respect to $x$? The chain rule of calculus is used to compute the derivatives of functions formed by composing other functions whose derivatives are known.

$$
\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx} \tag{3.1}
$$

### 4. Case Study

![backprop2.png](2022/09/07/iTFg9de8RyfJavm.png){: w="300" h="600" }

Let's first see an important example. In fully connected layers, one input neuron sends information (i.e., multiplied by weights) to every output neuron. Denote $w_{ji}$ as the weight from $x_i$ to $y_j$. Then for every output neuron (e.g., $y_j$), it accepts the information sent by every input neuron:

$$
y_{j}= \sum\limits_{i} w_{ji} x_{i} \tag{4.1}
$$

Then the partial derivative of $y_j$ with respect to $x_i$:

$$
\frac{\partial y_j}{\partial x_{i}}= w_{ji} \tag{4.2}
$$

Let's see another example. It is represented by the computational graph below.

![backprop1.png](2022/09/07/QOJYtIw6BNmsc2i.png){: w="300" h="600" }

And we can perform a forward propagation according to the computational graph.

$$
\begin{align}
h_{j} &= \sum\limits_{i} w_{ji}^{(1)} x_{i} \tag{4.3} \\
z_{j} &= f(h_{j})     \tag{4.4}      \\
y_{k} &= \sum\limits_{j}w_{kj}^{(2)} z_{j} \tag{4.5} 
\end{align}
$$

where

$$
f(h) = \tanh(h) = \frac{e^h - e^{-h}}{e^h + e^{-h}} \tag{4.6}
$$

A useful feature of this activation is that its derivative can be expressed in a particularly simple form:

$$
f'(h) = 1 - f(h)^2 \tag{4.7}
$$

The error function can be mean squared errors:

$$
E(\boldsymbol{w}) = \frac{1}{2} \sum\limits_{k}(y_{k}- \hat{y}_k)^2 \tag{4.8}            
$$

If we want to update the parameters, we need first to compute the partial derivative of $E(\boldsymbol{w})$ with respect to them. 

$$
\frac{\partial E(\boldsymbol{w})}{\partial w_{kj}^{(2)}} = \frac{\partial E(\boldsymbol{w})}{\partial y_{k}} \frac{\partial y_k}{\partial w_{kj}^{(2)}} = (y_{k}- \hat{y}_k)z_{j} \tag{4.9}
$$

$$
\begin{align}
\frac{\partial E(\boldsymbol{w})}{\partial w_{ji}^{(1)}} &= \frac{\partial E(\boldsymbol{w})}{\partial h_{j}}\frac{\partial h_j}{\partial w_{ji}^{(1)}} = (\frac{\partial E(\boldsymbol{w})}{\partial z_{j}} \frac{\partial z_j}{\partial h_j})x_{i} \tag{4.10} \\
\end{align}
$$

$$
\frac{\partial E(\boldsymbol{w})}{\partial z_j} = \sum\limits_{k}\frac{\partial E(\boldsymbol{w})}{\partial y_{k}}\frac{\partial y_k}{\partial z_{j}}= \sum\limits_{k} (y_{k}- \hat{y}_{k}) w_{kj}^{(2)}\tag{4.11}
$$

$\text{Remark.}$ $z_j$ can send information to all the output neurons (e.g., $y_k$), thus we need to sum over all the derivatives with respect to $z_j$.

Substituting $\text{(4.11)}$ into $\text{(4.10)}$ we obtain

$$
\frac{\partial E(\boldsymbol{w})}{\partial w_{ji}^{(1)}} = (1 - z_j^2)x_{i} \sum\limits_{k} (y_{k}- \hat{y}_{k}) w_{kj}^{(2)} \tag{4.12}
$$

### 5. Interpretation

Recall the Taylor approximation of the two variables function:

$$
f(x, y) = f(x_0, y_0) + f_x (x- x_0) + f_y(y-y_0) \tag{5.1}
$$

$\text{Remark.}$ $(x, y)$ needs to be close to $(x_0, y_0)$, otherwise the approximation can fail.

We can transform $\text{(5.1)}$ into $\text{(5.3)}$:

$$
\begin{align}
f(x,y) - f(x_{0},y_0) &= f_x (x- x_0) + f_y(y-y_0) \tag{5.2}\\
\implies \Delta f &= f_x \Delta x  + f_y \Delta y\tag{5.3}
\end{align}
$$

If we apply $\text{(5.3)}$ in the example above, we can obtain

$$
\Delta E(\boldsymbol{w}) = \nabla_{\boldsymbol{w}}E(\boldsymbol{w}) \Delta \boldsymbol{w} \tag{5.4}
$$

From another perspective, a small change in the parameters will propagate into a small change in object function by getting multiplied by the gradient.

> To summarize, back-propagation allows information to flow backwards through the network. This information can tell the model a small change in one particular parameter can result in what change in the object function. And gradient descent can use this information to adjust the parameters for optimizing the object function.
{: .prompt-tip }
