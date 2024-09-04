# DEVSOC_NN_IMAGE_CLASSFICATION
This repository is devoted to implementing neural networks for mnist image classification .

The code has been evaluated through the DIGIT RECOGNIZER competition on kaggle . 

link = https://www.kaggle.com/competitions/digit-recognizer/code?competitionId=3004&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false

Accuracy obtained = 92.810

-LINEAR LAYER CLASS-

A linear layer class is created that implements a fully connected (linear) layer.

The forward method computes the forward pass: y = Wx + b . 

-RELU ACTIVATION FUNC CLASS-

This class implements the ReLU activation function.

ReLU activation function goes like this: f(x) = max(0, x) .

-SIGMOID ACTIVATION FUNC CLASS-

This class implements the sigmoid activation function.

Sigmoid activation function goes like this: f(x) = 1 / (1 + e^(-x)).

-TANH ACTIVATION FUNC CLASS-

This class implements the tanh activation function.

Tanh activation function goes like this: f(x) = (e^x - e^(-x)) / (e^x + e^(-x)).

-SOFTMAX ACTIVATION FUNC CLASS-

This class implements the softmax activation function.

Softmax activation function goes like this: f(x_i) = exp(x_i) / Σ(exp(x_j)).

This activation function helps to calculate the probabilities of the indivisual labels and the highest probability becomes the label .

-SGD OPTIMIZER CLASS-

This class implements the Stochastic Gradient Descent optimization algorithm.

The SGD algorithm updates the parameters i.e weights and biases in the opposite direction of the gradient of the loss function with respect to the parameters.

The size of the step is determined by the learning rate.

## MODEL CLASS 
1. __init__ method:
   Initializes the model with empty lists for layers, loss function, and optimizer.
3. add_layer method:
   Adds a layer to the model's list of layers.
4. compile method:
   Sets the loss function and optimizer for the model.
5. forward method:
   Performs a forward pass through all layers in the model.
6. backward method:
   Performs a backward pass through all layers in reverse order.
7. train method:
   
   Trains the model for a specified number of epochs and batch size.

   Shuffles the data for each epoch.

   Performs forward and backward passes for each batch.

   Updates weights and biases using the optimizer.\

   Prints the average loss for each epoch.
8. predict method:
   Makes predictions on new data by performing a forward pass.
9. evaluate method:
    
   Evaluates the model's performance on a dataset.

   Calculates loss and accuracy.
   
10. save method:
    Saves the model's weights and biases to a file using pickle.
   
13. load method:
    Loads weights and biases from a file and assigns them to the appropriate layers.

    Handles potential mismatches between the saved weights and the model's current architecture.

-ONE HOT ENCODING AND NORMALIZING -

One - Hot encoding is done to make sure that the data is represented numerically
Features (X_train) are normalized pixel values.
Labels (y_train) are one-hot encoded representations of the digits.

-COMPILING AND TRAINING THE MODEL -

The model is created with 4 layers -

The first layer has the Linear function .

The second layer has ReLU func . 

The third layer has Linear func . 

And the last layer has the softmax function in order to calculate the probabilites of the digits . 

SGD Optimier is used with learning rate 0.01 ( feel free to experiment ) .

# SUBMISSION AND EVALUATION -
For submission a csv file is created that has the image id (index) and the predicted label of the test dataset . 

The model is saved and the paramters (weights and biases) are loaded . 

THANK YOU

AMAN PATEL 

BITS PILANI - GOA CAMPUS
