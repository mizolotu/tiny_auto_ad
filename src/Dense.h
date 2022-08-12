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

	private:

	public:

		Dense(short input_dim, short n_units);
    	virtual ~Dense();

    	float* forward(float *x);

};

#endif /* DENSE_H_ */
