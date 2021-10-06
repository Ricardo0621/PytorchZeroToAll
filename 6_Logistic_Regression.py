from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim

# Training data and ground truth
# x_data is the number of hours studied
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
# y_data is if it passes or not
y_data = tensor([[0.], [0.], [1.], [1.]])
hours_of_study = 7.0

class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        # Using a sigmoid function in our Linear model. Sigmoid is an activation function. An activation function defines an output given an input
        y_pred = sigmoid(self.linear(x))
        # Other activation functions: ReLU, ReLU6, ELU, SELU, PReLU, LeakyReLU, Threshold, Hardtanh, Sigmoid, Tanh
        # An activation function in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network.
        # Sometimes the activation function is called a “transfer function.” If the output range of the activation function is limited, 
        # then it may be called a “squashing function.” Many activation functions are nonlinear and may be referred to as the “nonlinearity” 
        # in the layer or the network design
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

# Defining a Binary Cross Entrophy function
loss_function = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = loss_function(y_pred, y_data)
    # print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

    # Inside the training loop, optimization happens in three steps:
    # Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
    optimizer.zero_grad()
    # Backpropagate the prediction loss with a call to loss.backwards(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
    loss.backward()
    # Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
    optimizer.step()

# After training
print(f'\nLet\'s predict the hours needed to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[hours_of_study]]))
print(f'Prediction after {hours_of_study} hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
