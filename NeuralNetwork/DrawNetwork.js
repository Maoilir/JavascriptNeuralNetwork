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

var svgns = "http://www.w3.org/2000/svg";
var xlinkns = "http://www.w3.org/1999/xlink";

function renderNN(_NN) {
	var pictureContainer = document.getElementById("pictureContainer");
	while (pictureContainer.firstChild) {
		pictureContainer.removeChild(pictureContainer.firstChild);
	}
	var _svg = document.createElementNS(svgns, "svg");

	var largestLayer = Math.max.apply(window, _NN.layers);

	centerLine = largestLayer * 100; //largestLayer/2*20
	for (var i = 0; i < _NN.layers.length; i++) {
		var layer = _NN.layers[i];
		var padding = (largestLayer - layer) * 100;
		var cx = i * 200 + 100;

		for (var j = 0; j < layer; j++) {
			var cy = padding + (200 * j) + 100;
			var circle = document.createElementNS(svgns, "circle");
			var circleColor = Math.floor(255 * _NN.Neurons[i][j].value);

			circle.cx.baseVal.value = cx;
			circle.cy.baseVal.value = cy;
			circle.r.baseVal.value = 5;
			circle.setAttributeNS(null, "style", "fill:rgb(" + circleColor + ",0,0)");

			_svg.appendChild(circle);

			for (var k in _NN.Neurons[i][j]._forwardConnections) {
				var conn = _NN.Neurons[i][j]._forwardConnections[k];
				var line = document.createElementNS(svgns, "line");
				var color = Math.floor(255 * conn.leftNeuron.value);
				var rgb = "rgb(" + color + "," + 0 + "," + 0 + ")";
				if (conn.weight < 0) {
					rgb = "rgb(" + 0 + "," + color + "," + color + ")";
				}

				line.x1.baseVal.value = cx;
				line.y1.baseVal.value = cy;

				line.setAttribute("id", conn.leftNeuron.id + ":" + conn.rightNeuron.id);
				line.setAttribute("style", "stroke:" + rgb + ";stroke-width:" + 2 * Math.abs(conn.weight));
				_svg.appendChild(line);
			}
			for (var l in _NN.Neurons[i][j]._reverseConnections) {
				var conn = _NN.Neurons[i][j]._reverseConnections[l];
				var line = _svg.getElementById(conn.leftNeuron.id + ":" + conn.rightNeuron.id);
				if (line) {
					line.x2.baseVal.value = cx;
					line.y2.baseVal.value = cy;
				}
			}
		}
	}

	pictureContainer.appendChild(_svg);
	pictureContainer.setAttribute("style", "height:" + largestLayer * 200 + "px;width:" + _NN.layers.length * 200 + "px;");
	_svg.setAttribute("style", "height:" + largestLayer * 200 + "px;width:" + _NN.layers.length * 200 + "px;");
}