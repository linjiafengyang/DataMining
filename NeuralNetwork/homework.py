import numpy as np
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from neuralnetwork import nn_model, predict

X = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [1, -1]])
X = X.T
print(X.shape)
Y = np.array([[1, 1, 0, 0, 0]])
print(Y.shape)
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 2, num_iterations = 5000, print_cost=True)
print(parameters)
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.flatten())
plt.title("Decision Boundary for hidden layer size " + str(2))
plt.show()
# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
