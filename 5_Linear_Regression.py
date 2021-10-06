import torch
from torch import nn
from torch import tensor

# x_data is the number of hours studied
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
# y_data is the number of points obtained
y_data = tensor([[3.0], [4.0], [5.0], [6.0]])
# e.g: 1 hour of study -> 2 points. 2 hours of study -> 4 points usw
# hours_of_study is the parameter we pass on. What we want to predict is our score
hours_of_study = 5.0

# Steps to create a Neural Network
# Step 1: Design the Model using classes
# Step 2: Define a loss function and optimizer
# Step 3: Train your model

class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        # Initializing the Model with the class
        super(Model, self).__init__()
        # torch.nn.Linear applies a Linear transformation. The first parameter is the size of each input sample. The second is the size of the output sample
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# Our model
model = Model()

# Construct our loss function and an Optimizer. 
# The call to model.parameters() in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
loss_function = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# for name in model.named_parameters():
#     print (name)
# Other optimizers: torch.optim.Adagrad, torch.optim.Adam, torch.optim.Adamax, torch.optim.ASGD, torch.optim.LBFGS, torch.optim.RMSprop, torch.optim.Rprop, torch.optim.SGD

# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    # Both y_pred and y_data are tensors
    loss = loss_function(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Inside the training loop, optimization happens in three steps:
    # Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
    optimizer.zero_grad()
    # Backpropagate the prediction loss with a call to loss.backwards(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
    loss.backward()
    # Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
    optimizer.step()


# What we are trying to do is predict the score or points we will get if we study hours_of_study
hour_var = tensor([[hours_of_study]])
y_pred = model(hour_var)
print(f'If we study {hours_of_study} hours we will get a score of {model(hour_var).data[0][0].item()}')