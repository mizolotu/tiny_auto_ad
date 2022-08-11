/*
 * Dense.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#ifndef DENSE_H_
#define DENSE_H_

class Dense {

	int input_dim;
	int n_units;

	private:

	public:

    	Dense();
    	virtual ~Dense();

    	float* forward(float *x);

};

#endif /* DENSE_H_ */
