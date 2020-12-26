import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


class MyNeuralNetwork:
	def __init__(self):
		# Make all the layers

		# All nd arrays are indexed with height by width. Coordinate first specifies row and then column
		# Holds an array of all the layers, a layer is defined by a synapse matrix of size (prev layer output size) x
		# (number of neurons in this layer)
		# The weight_(i,j) for a neuron i and its input from neuron j a layer above is listed at location (j,i)
		self.layers = []

	def forward_propagate(self, input):
		layer_outputs = [input]
		for layer_synapses in self.layers:
			layer_outputs.append(sigmoid(np.dot(layer_outputs[-1], layer_synapses)))

		return layer_outputs

	# Assume inputs are already scaled
	def fit(self, x_train: np.array, y_train, num_of_hidden_layers=2, hidden_neurons_in_layer=5, epochs=10000,
			learning_rate=0.1, batch_size=100):
		if batch_size > x_train.shape[0]:
			batch_size = x_train.shape[0]

		# Build the layers
		last_output_size = x_train.shape[1]
		for layer_i in range(num_of_hidden_layers):
			new_layer = 2 * np.random.random((last_output_size, hidden_neurons_in_layer)) - 1
			self.layers.append(new_layer)
			last_output_size = hidden_neurons_in_layer

		# Make sure that the ndarrays we're working with are 2D
		# Build output layer
		try:
			num_outputs = y_train.shape[1]
		except:
			num_outputs = 1
			y_train = np.reshape(y_train, (-1, 1))
		output_layer = 2 * np.random.random((last_output_size, num_outputs))
		self.layers.append(output_layer)

		# Now train the model
		# Train it!
		for training_iter in range(epochs):
			# Do it in batches
			selected_batch = np.random.choice(x_train.shape[0], size=batch_size, replace=False)
			x_batch = x_train[selected_batch, :]
			y_batch = y_train[selected_batch, :]
			# if num_outputs == 1:
			# 	y_batch = y_train[selected_batch]
			# else:
			# 	y_batch = y_train[selected_batch, :]

			layer_outputs = self.forward_propagate(x_batch)
			y_pred = layer_outputs[-1]

			# Calculate the cost
			# cost = 0.5 * np.power(y_batch - y_pred, 2)
			cost = (1 / (2 * batch_size)) * np.power(y_pred - y_batch, 2)
			# Sum all costs for a single output neuron together
			cost = np.sum(cost, axis=0)
			if training_iter % 1000 == 0:
				print("Cost at " + str(training_iter) + " is " + str(cost))
			# print("Layer 0: \n" + str(layers[0]))

			# Store all of the partial derivatives of cost with respect to the output of a neuron for every layer

			# Generate all the cost partials
			# The first partial for the last layer is
			cost_partial_for_neurons = [1 / (num_outputs * batch_size) * (y_pred - y_batch)]
			# Sum all the different samples together
			# np.sum(cost_partial_for_neurons, axis=0)
			# cost_partial_for_neurons = [-1 * (y_batch - y_pred)]

			# Cost partial for neurons should always maintain that the first element's matrix is of shape
			# (batch_size) x (neurons in layer)

			# Each following one is dependent on the first
			for layer_i in range(len(self.layers) - 2, -1, -1):
				modded_ak = np.multiply(np.multiply(layer_outputs[layer_i + 2], (1 - layer_outputs[layer_i + 2])),
										cost_partial_for_neurons[0])
				# FLAWED: Doesn't accurately account for if we are doing this with more than one observation in the
				# training batch
				cost_partial_for_layer_neurons = np.dot(self.layers[layer_i + 1], modded_ak.T).T
				# cost_partial_for_layer_neurons = np.sum(cost_partial_for_layer_neurons, axis=1)
				cost_partial_for_neurons.insert(0, cost_partial_for_layer_neurons)

			cost_partial_for_weights = []

			# Go through all layers, calculate how much the synapses (weights) contributed to the error.
			for layer_i in range(0, len(self.layers)):
				# Generate partials
				# layer_i is the index of the layer and the outputs from a layer are at index layer_i + 1 in
				# layer_outputs
				modded_a = np.multiply(np.multiply(layer_outputs[layer_i + 1], (1 - layer_outputs[layer_i + 1])),
									   cost_partial_for_neurons[layer_i])

				# Each entry in cost partial for weights is going to be a 3D array with shape (batch_size) x (num
				# neurons layer j) x (num neurons layer i)
				cost_partial_for_weights.append(np.zeros((batch_size, self.layers[layer_i].shape[0], self.layers[
					layer_i].shape[1])))
				for t in range(batch_size):
					# Take the outputs from neuron in the layer above (layer_outputs[layer_i][t]) and scalar multiply
					# with each output from this layer and append all of the new rows under each other
					layer_above_outputs_t = np.reshape(layer_outputs[layer_i][t, :], (1, -1))
					layer_outputs_t = np.reshape(modded_a[t, :], (1, -1))

					cost_partial_for_weights[layer_i][t] = np.matmul(layer_above_outputs_t.T, layer_outputs_t)

				#cost_partial_for_weights.append(np.dot(layer_outputs[layer_i].T, modded_a))

				# Take the average update from each sample
				cost_partial_for_weights[layer_i] = np.sum(cost_partial_for_weights[layer_i], axis=0) / batch_size

				# Update the weights
				self.layers[layer_i] -= learning_rate * cost_partial_for_weights[layer_i]

		# cost_partial_for_weights now contains the (partial E) / (partial w_(i,j)) for all neurons i in a layer
		# and their input neuron j a layer above
		# Neurons i go across by column, input neurons i are listed down the rows
		# The (partial E) / (partial w_(i,j)) is at location (j,i)

		print("Finished Training")
		print("Cost at end: " + str(cost))
