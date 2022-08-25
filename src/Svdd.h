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
	short c_n;
	bool is_c_fixed;
	float learning_rate;
	float score;
	int score_n;
	float score_sum;
	float score_ssum;
	float score_thr;
	bool is_score_thr_fixed;
	short score_alpha;

	public:

		Svdd(short n, Dense *l, float lr, short a);
		virtual ~Svdd();

		short get_output_dim();
		void forward(float* x);
		void fix_c();
    	void backward(float* x);
    	float get_c(short i);
    	float get_score();
    	void switch_score_thr();
    	float get_score_thr();

};

#endif /* SVD_H_ */
