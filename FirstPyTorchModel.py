import torch
import matplotlib.pyplot as plt
from torch import nn

# Create our parameters
# We don't normally know this info but for the our first model we'll create this our own
weight = 0.3
bias = 0.9

# Create our tensor values
start = 0
end = 1
step = 0.01

# This is our input
# We unsqueeze this tensor to add a dimension
X = torch.arange(start, end, step).unsqueeze(dim=1)

# Our model will learn via Linear Regression Formula
# Linear Regression Formula: y = a + bX (a = bias) (weight = b) X = our tensor
y = weight * X + bias

" Split our data "
train_split = int(0.8 * len(X)) # We're trying to get 80% of our data as training data so we multiply by .8

# Gather 80% of our training data to use for training (the data we use to train an algorithm or ML model to predict the outcome we design our model to predict)
X_train, y_train = X[:train_split], y[:train_split]

# This is our test data (Once our ML model is built (with our training data), we need to unsee data to test our model)
X_test, y_test = X[train_split:], y[train_split:]

# We're going to use the matplotlib module to visualize our data
" Building a function to visualize our data "
def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_label=y_test, predictions=None):

    # The size of the window of our graph window
    plt.figure(figsize=(10,10))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training label')

    # Plot testing data in red
    plt.scatter(test_data, test_label, c='r', s=4, label='Testing label')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='g', s=4, label='Predictions')

" Create our first PyTorch model in Linear Regression "
# Every model we'll every make needs to import from nn.Module
class LinearRegressionModel(nn.Module):
    # Create a constructor 
    def __init__(self):
        super().__init__()
        
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
# Creating instance of our model
model_0 = LinearRegressionModel()

# Check what kind of predictions our model is making before we start to train it 
with torch.inference_mode():
    # We're passing our X_test data through our model
    y_preds = model_0(X_test)

" Setting Up an Optimizer and a Loss Function "
# loss function
loss_fn = nn.L1Loss()

# The optimizers objective is to give the model values, so parameters like wieghts and bias that minimize the loss function 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) # lr = learning rate (Smaller = smaller change to data, Bigger = bigger change to data)

" PyTorch Training Loop Steps and Intuition "
torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    
    # Set the model to training mode 
    model_0.train()

    # 1. Forward pass
    # We learn patterns on the training data to evaluate our model on the test data
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    # We're calculating the different in our mdoels predictions on the training dataset and the ideal training dataset values
    # Predictions first, ideal training values second
    # y_pred (what we're training) y_train (what we want)
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    # We set this to zero because we don't want to keep track of all our data
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()
     
    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()

    # Training 
    model_0.eval()
    with torch.inference_mode():
        # 1. Do the forward pass 
        test_pred = model_0(X_test)

        # 2. Calculate the loss (test loss because it's on the test dataset)
        test_loss = loss_fn(test_pred, y_test)

# Get predictions for our trained model
with torch.inference_mode():
    y_preds_new = model_0(X_test)


# Display graph 
plot_predictions(predictions=y_preds_new)
plt.show()




































