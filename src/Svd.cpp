/*
 * Svd.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <cstdlib>
#include <math.h>
#include <algorithm>

#include "Dense.h"

using namespace std;

#include "Svd.h"

#include <iostream>

Svd::Svd(short n, Dense *l) {
	n_layers = n;
	layers = l;
	n_output = layers[n_layers - 1].get_output_dim();
	c = new float[n_output];
	for (short i=0; i < n_output; i++) {
		c[i] = 0.0;
	}
	is_c_fixed = false;
	n_c = 0;
}

Svd::~Svd() {
	// TODO Auto-generated destructor stub
}

short Svd::get_output_dim() {
	return n_output;
}

float* Svd::get_output() {
	return layers[n_layers - 1].get_output();
}

void Svd::fix_c() {
	for (short i=0; i<n_output; i++) {
		c[i] /= n_c;
	}
	is_c_fixed = true;
}

void Svd::forward(float *x) {
	float *layer_output;
	layer_output = x;
	for (short i=0; i<n_layers; i++) {
		layers[i].forward(layer_output);
		layer_output = layers[i].get_output();
	}
	if (is_c_fixed == false) {
		n_c += 1;
		for (short i=0; i<n_output; i++) {
			c[i] += layer_output[i];
		}
	}
}
