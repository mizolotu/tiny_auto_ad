/*
 * Svd.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "Dense.h"

#ifndef SVD_H_
#define SVD_H_

class Svd {

	short n_layers;
	Dense *layers;
	short n_output;
	float *c;
	short n_c;
	bool is_c_fixed;

	public:

		Svd(short n, Dense *l);
		virtual ~Svd();

		short get_output_dim();
		float* get_output();
		void forward(float *x);
		void fix_c();

};

#endif /* SVD_H_ */
