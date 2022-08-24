/*
 * Svd.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Dense.h"

#ifndef SVD_H_
#define SVD_H_

class Svdd {

	short n_layers;
	Dense *layers;
	short n_output;
	float *c;
	float score;
	short n_c;
	bool is_c_fixed;
	float learning_rate;

	public:

		Svdd(short n, Dense *l, float lr);
		virtual ~Svdd();

		short get_output_dim();
		void forward(float* x);
		void fix_c();
    void backward(float* x);
    float get_c(short i);
    float get_score();

};

#endif /* SVD_H_ */
