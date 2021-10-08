from torch import nn, optim, from_numpy
import numpy as np


# Load the file
xy = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
# Both x_data and y_data are Tensors
# Takes everything ut the last column
x_data = from_numpy(xy[:, 0:-1]) 
# Takes the last column
y_data = from_numpy(xy[:, [-1]])
# print(y_data)
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


# The vanishing gradient problem occurs when an activation function is added and the loss gradient tends to zero while certain activation functions are added to the neural networks

class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        # Just like in vectors, the number of columns of the first vector must be equal to the number of rows of the second vector
        #Anm, Bcd -> So can be multiplied if m==c. The resulting matrix will be of size Fnd
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)
        # The output is a 8x1 matrix

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


# our model
model = Model()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the three
# nn.Linear modules which are members of the model.
# BCE -> Binary cross enthropy
loss_function = nn.BCELoss(reduction='mean')
# Trying out the different optimizers
# 0.6447 SGD vs 0.4456 Rprop vs 0.4351 Adam
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# Rprop stands for Resilient Propagation. Rprop is a popular gradient descent algorithm that only uses the signs of gradients to compute updates
# optimizer = optim.Rprop(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.1)
# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = loss_function(y_pred, y_data)
    print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
