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
	float *weights;
	float *biases;
	float *output;
	float (*activation)(float);

	private:

	public:

		Dense();
		Dense(short input_dim, short n_units, float (*f)(float));
    	virtual ~Dense();

    	short get_output_dim();
    	float* get_output();
    	void forward(float *x);
    	void backward(float *y, float *l);

};

#endif /* DENSE_H_ */
