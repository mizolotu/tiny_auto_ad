/*
 * Dense.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Dense.h"
#include <math.h>

Dense::Dense() {}

Dense::Dense(short input_dim_, short n_units_, float (*f)(float)) {
	input_dim = input_dim_;
	n_units = n_units_;
	activation = f;
	weights = new float[input_dim * n_units];
	biases = new float[n_units];
	output = new float[n_units];
	for (short i=0; i < input_dim * n_units; i++) {
		//weights[i] = (random() % 1000) / 1000.0;
		weights[i] = 1.0;
	}
	for (short i=0; i < n_units; i++) {
		//biases[i] = (random() % 1000) / 1000.0;
		biases[i] = 1.0;
		output[i] = 0.0;
	}
}

Dense::~Dense() {
	delete[] weights;
	delete[] biases;
}

short Dense::get_output_dim() {
	return n_units;
}

float* Dense::get_output() {
	return output;
}

void Dense::forward(float *x) {
	for (int j = 0; j < n_units; j++) {
		output[j] = 0;
	    for (int i = 0; i < input_dim; i++) {
	    	output[j] += x[i] * weights[i * n_units + j];
	    }
	    output[j] = activation(output[j] + biases[j]);
	}
}

