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

#include "Svdd.h"

#include <iostream>

Svdd::Svdd(short n, Dense* l, float lr, short a) {
	n_layers = n;
	layers = l;
	n_output = layers[n_layers - 1].get_output_dim();
	c = new float[n_output];
	for (short i=0; i < n_output; i++) {
		c[i] = 0.0;
	}
	is_c_fixed = false;
	c_n = 0;
	learning_rate = lr;
	score = 0.0;
	score_n = 0;
	score_sum = 0.0;
	score_ssum = 0.0;
	score_thr = 0;
	is_score_thr_fixed = true;
	score_alpha = a;
}

Svdd::~Svdd() {}

short Svdd::get_output_dim() {
	return n_output;
}

void Svdd::fix_c() {
	for (short i=0; i<n_output; i++) {
		c[i] /= c_n;
		cout << "c[" << i << "] = " << c[i] << endl;
	}
	is_c_fixed = true;
}

float Svdd::get_c(short i) {
	return c[i];
}

float Svdd::get_score() {
	return score;
}

void Svdd::switch_score_thr() {
	cout << score_sum << endl;
	cout << score_n << endl;
	cout << score_ssum << endl;
	score_thr = score_sum / score_n + score_alpha * sqrt((max(0.0, score_ssum - score_n * pow(score_sum / score_n, 2))) / score_n);
	is_score_thr_fixed = !is_score_thr_fixed;
}

float Svdd::get_score_thr() {
	return score_thr;
}

void Svdd::forward(float* x) {
	float* layer_output;
	layer_output = x;
	for (short i=0; i<n_layers; i++) {
		layers[i].forward(layer_output);
		layer_output = layers[i].get_outputs();
	}

	if (is_c_fixed) {

		score = 0;
		for (short i=0; i<n_output; i++) {
			score += pow(c[i] - layer_output[i], 2);
		}

	} else {

		c_n += 1;
		for (short i=0; i<n_output; i++) {
			c[i] += layer_output[i];
		}

	}

	if (is_score_thr_fixed) {

	} else {

		score_n += 1;
		score_sum += score;
		score_ssum += pow(score, 2);

	}

}

void Svdd::backward(float* x) {

	// Last layer errors

	for (short i=0; i<n_output; i++) {
		layers[n_layers - 1].set_errors(i, c[i] - layers[n_layers - 1].get_outputs(i));
	}

	// Other layer errors

	float e;
	for (short l=n_layers-2; l>=0; l--) {
		for (short i=0; i<layers[l].get_output_dim(); i++) {
			e = 0;
			for (short j=0; j<layers[l + 1].get_output_dim(); j++) {
				e += layers[l + 1].get_weights(i, j) * layers[l + 1].get_errors(j);
			}
			layers[l].set_errors(i, e);
		}
	}

	// Update weights of the first layer

	for (short i=0; i<layers[0].get_input_dim(); i++) {
		for (short j=0; j<layers[0].get_output_dim(); j++) {
			layers[0].set_weights(layers[0].get_weights(i, j) + learning_rate * layers[0].get_errors(j) * layers[0].get_d_outputs(j) * x[i], i, j);
		}
	}

	// Update biases of the first layer

	for (short j=0; j<layers[0].get_output_dim(); j++) {
		layers[0].set_biases(layers[0].get_biases(j) + learning_rate * layers[0].get_errors(j) * layers[0].get_d_outputs(j), j);
	}

	// Update the rest of the layers

	for (short l=1; l<n_layers; l++) {

		// Update weights

		for (short i=0; i<layers[l].get_input_dim(); i++) {
			for (short j=0; j<layers[l].get_output_dim(); j++) {
				layers[l].set_weights(layers[l].get_weights(i, j) + learning_rate * layers[l].get_errors(j) * layers[l].get_d_outputs(j) * layers[l-1].get_outputs(i), i, j);
			}
		}

		// Update biases

		for (short j=0; j<layers[l].get_output_dim(); j++) {
			layers[l].set_biases(layers[l].get_biases(j) + learning_rate * layers[l].get_errors(j) * layers[l].get_d_outputs(j), j);
		}
	}

}
