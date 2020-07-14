# Tests backprop with a variable number of layers

import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


num_of_hidden_layers = 2
hidden_neurons_in_layer = 3
input_size = 3
output_size = 1
learning_rate = 0.05

X = np.array([
	[0, 1, 0.5],
	[0, 1, 0.5],
	[1, 0, -0.5],
	[1, 0, -0.5]
])

Y = np.array([
	[1],
	[1],
	[0],
	[0]
])

# All nd arrays are indexed with height by width. Coordinate first specifies row and then column
# Holds an array of all the layers, a layer is defined by a synapse matrix of size (prev layer output size) x (number
# of neurons in this layer)
# The weight_(i,j) for a neuron i and its input from neuron j a layer above is listed at location (j,i)
layers = []

# Build the layers
last_output_size = input_size
for layer_i in range(num_of_hidden_layers):
	new_layer = 2 * np.random.random((last_output_size, hidden_neurons_in_layer)) - 1
	layers.append(new_layer)
	last_output_size = hidden_neurons_in_layer

# Build output layer
output_layer = 2 * np.random.random((last_output_size, output_size))
layers.append(output_layer)


# input is an input nd array with width = input_size. Returns outputs of all layers, last layer is the output
def forward_propagate(input):
	layer_outputs = [input]
	for layer_synapses in layers:
		layer_outputs.append(sigmoid(np.dot(layer_outputs[-1], layer_synapses)))

	return layer_outputs


# Returns an array of the cost partials for all layers from layer_i and beyond
# layer_i is positive indexed starting at 0 and going up to the len(layers)
def cost_partials_for_neuron_layer(layer_i):
	if layer_i == len(layers) - 1:
		# The output layer
		return [-1 * (Y - Y_pred)]
	else:
		previous_partials = cost_partials_for_neuron_layer(layer_i + 1)
		modded_ak = layer_outputs[layer_i + 1] * (1 - layer_outputs[layer_i + 1]) * previous_partials[0]
		cost_partial_for_layer_neurons = np.dot(layers[layer_i], modded_ak.T).T
		# cost_partial_for_layer_neurons = np.sum(cost_partial_for_layer_neurons, axis=1)
		previous_partials.insert(0, cost_partial_for_layer_neurons)
		return previous_partials


for training_iter in range(10000):
	layer_outputs = forward_propagate(X)
	Y_pred = layer_outputs[-1]

	# Calculate the cost
	cost = 0.5 * np.power(Y - Y_pred, 2)
	# Sum all costs for a single output neuron together
	cost = np.sum(cost, axis=0)
	if training_iter % 1000 == 0:
		print("Cost at " + str(training_iter) + " is " + str(cost))
	# cost = np.sum(cost, axis=1)

	# Generalize it for all layers
	# Store all of the partial derivatives of cost with respect to the output of a neuron for every layer

	cost_partial_for_neurons = cost_partials_for_neuron_layer(0)
	cost_partial_for_weights = []

	# Go through all layers, calculate how much the synapses (weights) contributed to the error.
	for layer_i in range(0, len(layers)):
		# Generate partials
		# layer_i is the index of the layer and the outputs from a layer are at index layer_i + 1 in layer_outputs
		modded_a = layer_outputs[layer_i + 1] * (1 - layer_outputs[layer_i + 1]) * cost_partial_for_neurons[layer_i]
		cost_partial_for_weights.append(np.dot(layer_outputs[layer_i].T, modded_a))

		# Update the weights
		layers[layer_i] -= learning_rate * cost_partial_for_weights[layer_i]

	# cost_partial_for_weights now contains the (partial E) / (partial w_(i,j)) for all neurons i in a layer
	# and their input neuron j a layer above
	# Neurons i go across by column, input neurons i are listed down the rows
	# The (partial E) / (partial w_(i,j)) is at location (j,i)


print("Cost at end: " + str(cost))

print("Pred output: \n" + str(forward_propagate(X)[-1]))

