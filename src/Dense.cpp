/*
 * Dense.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Dense.h"
#include <math.h>

Dense::Dense(short input_dim_, short n_units_) {
	input_dim = input_dim_;
	n_units = n_units_;
	weights = new float[input_dim * n_units];
	for (short i=0; i < input_dim * n_units; i++) {
		weights[i] = random();
	}

}

Dense::~Dense() {
	// TODO Auto-generated destructor stub
}

