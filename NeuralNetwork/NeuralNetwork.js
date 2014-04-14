//A framework for creating neural networks in javascript
//Copyright (C) 2014 Ryan Browne <maoilir@gmail.com>

//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.

learningRate = .10;
momentum = .1;
log = function(a) {
	console.log(a);
}

NeuralNetwork = function(layers, options) {
	if (layers && layers.length) {
		this.layers = layers;
		this.Neurons = new Array();
		this.Biases = new Array();
		for (var i = 0; i < layers.length; i++) {
			var nPos = this.Neurons.push(new Array(layers[i])) - 1;
			if (i > 0) {
				var l = this.Biases.push(new Neuron(i, -1));
				this.Biases[l - 1].setValue(1);
			}
			for (var j = 0; j < layers[i]; j++) {
				this.Neurons[nPos][j] = new Neuron(i, j);
				if (i > 0) {
					for (var k = 0; k < layers[nPos - 1]; k++) {
						Neuron.connect(this.Neurons[nPos - 1][k], this.Neurons[nPos][j], Math.random());
					}
					Neuron.connect(this.Biases[i - 1], this.Neurons[nPos][j], Math.random());
				}
			}
		}
	}
}
NeuralNetwork.prototype = {
	run: function(inputs) {
		this.reset();
		//log("Running pattern: " + inputs.toString());
		for (var i = 0; i < this.layers[0]; i++) {
			// log("Setting Neuron " + i + " to " + inputs[i]);
			this.Neurons[0][i].setValue(inputs[i]);
		}
		return this.saveOutput();
	},
	train: function(inputs, outputs, allowedError) {
		var totalError = 0;
		var errors = new Array();
		this.run(inputs, false);
		for (var i = 0; i < this.layers[this.layers.length - 1]; i++) {
			var error = outputs[i] - this.Neurons[this.layers.length - 1][i].value;
			errors.push(error);
			totalError += Math.abs(error);
		}

		totalError = totalError / i;

		for (var j = 0; j < this.layers[this.layers.length - 1]; j++) {
			this.Neurons[this.layers.length - 1][j].setError(errors[j]);
		}
		for (var k = 0; k < this.layers[this.layers.length - 1]; k++) {
			this.Neurons[this.layers.length - 1][k].backPropagate();
		}

		return totalError;
	},
	trainAll: function(data, allowedError) {
		var cont = true;
		for (var i = 0; i < 1000 && cont; i++) {
			var totalError = 0;
			for (var j = 0; j < data.length; j++) {
				var error = this.train(data[j][0], data[j][1], allowedError);
				totalError += error;
			}
			cont = totalError / data.length > allowedError;
		}
		return !cont;
	},
	saveOutput: function() {
		var returned = new Array();
		for (var j = 0; j < this.layers[this.layers.length - 1]; j++) {
			returned.push(this.Neurons[this.layers.length - 1][j].value);
		}
		this.output = returned;
		return returned;
	},
	printOutput: function() {
		return this.output;
	},
	reset: function() {
		for (var i = 0; i < this.layers.length; i++) {
			for (var j = 0; j < this.layers[i]; j++) {
				this.Neurons[i][j].reset();
			}
		}
		for (var k = 0; k < this.Biases.length; k++) {
			this.Biases[k].resetError();
		}
	},
	jsonEncode: function() {
		return JSON.stringify(JSON.decycle(this));
	},
};
//TODO: Not quite. We also need to hook back up their prototypes.
//TODO: Needs to be tested.
NeuralNetwork.jsonImport = function(json) {
	var NN = JSON.retrocycle(JSON.parse(json));
	NN.constructor = NeuralNetwork;
	NN.prototype = NeuralNetwork.prototype;
	for (var i = 0; i < NN.layers.length; i++) {
		for (var j = 0; j < NN.layers[i]; j++) {
			NN.Neurons[i][j].constructor = Neuron;
			NN.Neurons[i][j].prototpye = Neuron.prototype;

			for (var k in NN.Neurons[i][j]._forwardConnections) {
				var conn = NN.Neurons[i][j]._forwardConnections[k];
				conn.constructor = Connection;
				conn.prototype = Connection.prototype;
			}
		}
	}
	return NN;
}

Neuron = function(layer, position) {
	this.id = layer + "-" + position;
	this._forwardConnections = {};
	this._reverseConnections = {};
	this._value = null;
	this._error = null;
}
Neuron.prototype = {
	createForwardConnection: function(connection) {
		this._forwardConnections[connection.rightNeuron.id] = connection;
	},
	createReverseConnection: function(connection) {
		this._reverseConnections[connection.leftNeuron.id] = connection;
	},
	setValue: function(value) {
		this._value = value || 0;
		//log("Neuron " + this.id + " now has a value of " + this._value);
	},
	setError: function(error) {
		this._error = error;
	},
	determineValue: function() {
		var value = 0;
		for (var i in this._reverseConnections) {
			value += this._reverseConnections[i].value;
		}
		// log("Neuron " + this.id +" sees the values from previous neurons as: " + value);
		this._value = 1 / (1 + Math.exp(-value));
		// log("Neuron " + this.id + " now has a value of " + this._value);
	},
	determineError: function() {
		var error = 0;
		for (var i in this._forwardConnections) {
			error += this._forwardConnections[i].error;
		}
		// log("Neuron " + this.id + " sees the error from previous neurons as: " + error);
		this._error = this._value * (1 - this._value) * error;
	},
	forwardPropagate: function() {
		// log("Forward propagating neuron " + this.id + " currently at " + this.value);
		for (var i in this._forwardConnections) {
			this._forwardConnections[i].forwardPropagate();
		}
	},
	backPropagate: function() {
		for (var i in this._reverseConnections) {
			this._reverseConnections[i].backPropagate();
		}
	},
	reset: function() {
		this._value = null;
		this._error = null;
		for (var i in this._forwardConnections)
			this._forwardConnections[i].reset();
	},
	resetError: function() {
		this._error = null;
		for (var i in this._forwardConnections)
			this._forwardConnections[i].reset();
	},
	get value() {
		if (this._value === null)
			this.determineValue();
		return this._value;
	},
	get error() {
		if (this._error === null)
			this.determineError();
		return this._error;
	}
};
Neuron.connect = function(n1, n2, weight) {
	var connection = new Connection(n1, n2, weight);
}

Connection = function(n1, n2, weight) {
	this.leftNeuron = n1;
	this.rightNeuron = n2;
	this.weight = weight;
	this.previousChange = 0;

	n1.createForwardConnection(this);
	n2.createReverseConnection(this);
}
Connection.prototype = {
	reset: function() {
		this.previousChange = 0;
	},
	forwardPropagate: function() {
		this.rightNeuron.forwardPropagate();
	},
	backPropagate: function() {
		var currentWeight = this.weight;
		var change = this.rightNeuron.error * this.leftNeuron.value * learningRate;
		this.weight += change;
		this.previousChange = change;
		// log("Connection from " + this.leftNeuron.id + " to " + this.rightNeuron.id + " is adjusting its weight to " + this.weight);
		this.leftNeuron.backPropagate();
	},
	get error() {
		// log("Connection from " + this.leftNeuron.id + " to " + this.rightNeuron.id + " has a weight of " + this.weight + " and an outgoing error of " + this.rightNeuron.error);
		return this.rightNeuron.error * this.weight;
	},
	get value() {
		// log("Connection from " + this.leftNeuron.id + " to " + this.rightNeuron.id + " has a weight of " + this.weight + " and an incoming value of " + this.leftNeuron.value);
		return this.leftNeuron.value * this.weight;
	}
};