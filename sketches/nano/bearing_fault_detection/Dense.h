/*
 * Dense.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#ifndef DENSE_H_
#define DENSE_H_

class Dense {

	short input_dim;
	short n_units;
	float (*activation)(float);
	float (*d_activation)(float);
	float *weights;
	float *biases;
	float *outputs;
	float *errors;

	private:

	public:

		Dense();
		Dense(short input_dim, short n_units, float (*f)(float), float (*df)(float));
    	virtual ~Dense();

    	short get_input_dim();
    	short get_output_dim();
    	float* get_outputs();
    	void forward(float *x);
    	void set_errors(short i, float e);
    	float* get_errors();
    	float get_weights(short i, short j);
    	float set_weights(float w, short i, short j);
    	float get_biases(short i);
    	float set_biases(float b, short i);
    	float get_outputs(short i);
    	float get_errors(short i);
    	float get_d_outputs(short i);

};

#endif /* DENSE_H_ */
