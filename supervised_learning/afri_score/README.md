# AFRISCORE
Afriscore is a machine learning project that leverages machine learning techniques to develop a credit scoring algorithm that determines whether or not individuals are worthy of receiving credit from their Banks.
A detailed proposal to this project can be found [here;](https://docs.google.com/document/d/1XpOUKyt2DH6wQ9Pfeq8w31DLfAjhRlRRcslEcjtVSMM/edit?usp=sharing)

# Afriscore Lean canvas
A lean canvas that defines the business/economic relevance of Afriscore can be found [here;](https://docs.google.com/presentation/d/1CuriQnHya5FZHXnSRDSemWuFRwSQiF11ElNvd3cSdUw/edit?usp=sharing)

## Long Short-Term Memory (LSTM)

Beyond the typical Neural networks, Afriscore employs Long Short-Term Memory (LSTM).

LSTM is a type of recurrent neural network (RNN) that is capable of learning long-term dependencies in sequence data. This is particularly useful in many real-world tasks because many sequences (such as sentences, time series data, etc.) have dependencies over time.

Traditional RNNs have difficulties learning these long-term dependencies due to the "vanishing gradients" problem, where the contribution of information decays geometrically over time. LSTMs overcome this problem with a unique design.

An LSTM has a similar control flow as a standard RNN, it processes data by passing through a sequence of gates, including a forget gate, input gate, and output gate. However, it also has a cell state that runs along the entire chain, with only minor linear interactions. This design helps it to keep or forget information effectively.

In this project, LSTM is used as part of the model architecture. The LSTM layer takes in the sequence data and outputs a sequence with the same length, which can be fed into the next layer in the model. This allows the model to understand the temporal dependencies in the sequence data, which is crucial for the credit scoring task.

The main parameters for the LSTM layer are the number of LSTM units and the input shape. The number of LSTM units is the dimensionality of the output space, which can be tuned based on the complexity of the task. The input shape is usually the shape of the sequence data.

In this project, 50 LSTM units are used. This value was chosen based on empirical results showing that it provides a good balance between model complexity and performance.

## Optimization Techniques

In this project, several optimization techniques are used to train the deep learning models. These techniques are crucial for finding the best set of parameters that minimize the loss function and improve the model's performance.

### Stochastic Gradient Descent (SGD)

SGD is a variant of gradient descent, a popular optimization algorithm in machine learning. Instead of using the entire data set to compute the gradient, SGD uses a single random example at each iteration. This makes SGD faster and able to handle large datasets.

The main parameter for SGD is the learning rate, which controls the step size during the gradient descent process. A smaller learning rate means the model will learn slowly, which can lead to better performance but also longer training times. A larger learning rate means the model will learn quickly, but it might also overshoot the optimal solution.

In the project herein, a learning rate of 0.01 was used to provive optimum learning of the model.

It is important to note that one of the challenges with SGD is the vanishing gradient where the gradients of the loss function become vert small as they are backpropagated through layers of the network. Since the gradients are small, the updates to the weights become smaller too, resulting in signficantly slower training and eventual ceasation of the network's learning.

### RMSprop

RMSprop uses a moving average of squared gradients to normalize the gradient itself. This helps to resolve the issue of diminishing learning rates experienced in SGD.

The main parameters for RMSprop are the learning rate and the decay factor. The learning rate is similar to the one in SGD. The decay factor controls the rate at which the moving average decays, similar to momentum.

In this project, a learning rate of 0.001 and a decay factor of 0.9 for RMSprop were used.

## Conclusion

Optimization techniques and parameter tuning play a crucial role in the development of machine learning models. By understanding the underlying principles of these techniques and carefully tuning the parameters, the performance of a model is significantly improved.