JavascriptNeuralNetwork
=======================

An object-oriented framework for building backpropagating neural networks in javascript.

Usage:
Copy the NeuralNetwork folder to your application, and add

\<script src="NeuralNetwork/NeuralNetwork.js"></script>

For serialization and de-serialization, you will need

\<script src="vendor/JSON/cycle.js"></script>

from https://github.com/douglascrockford/JSON-js

To create a new neural network with 3 layers, with 4 neurons, 5 neurons, and 3 neurons, in the respective layers, use

var nn = new NeuralNetwork([4,5,3]);

To get a visualization of the network, include

\<script src="NeuralNetwork/DrawNetwork.js"></script>

and call

renderNN(nn);
