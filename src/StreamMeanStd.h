/*
 * StreamMeanStd.h
 *
 *  Created on: Aug 25, 2022
 *      Author: mizolotu
 */

#ifndef STREAMMEANSTD_H_
#define STREAMMEANSTD_H_

class StreamMeanStd {

	short dimension;     // dimension
	short count;         // number of elements
	float *sum;          // sum of elements
	float *ssum;         // sum of element squares

	private:

	public:

		StreamMeanStd(short dim);
		~StreamMeanStd();

		void enqueue(float *x);       // add a vector
		void mean(float* m);          // calculate mean
		void std(float* s);           // calculate std

};

#endif /* STREAMMEANSTD_H_ */
