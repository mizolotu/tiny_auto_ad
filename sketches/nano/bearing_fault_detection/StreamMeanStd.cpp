/*
 * StreamMeanStd.cpp
 *
 *  Created on: Aug 25, 2022
 *      Author: mizolotu
 */

#include <cstdlib>
#include <math.h>
#include <algorithm>

#include "StreamMeanStd.h"

using namespace std;

StreamMeanStd::StreamMeanStd(short dim) {
	dimension = dim;
	count = 0;
	sum = new float[dim];
	ssum = new float[dim];
	for (short i=0; i < dim; i++) {
		sum[i] = 0;
		ssum[i] = 0;
	}

}

StreamMeanStd::~StreamMeanStd() {
	delete[] sum;
	delete[] ssum;
}

void StreamMeanStd::enqueue(float *x) {

    for (short i=0; i < dimension; i++) {
    	sum[i] += x[i];
    	ssum[i] += pow(x[i], 2);
    }

    count++;

}

void StreamMeanStd::mean(float* m) {
	for (short i = 0; i < dimension; i++) {
		m[i] = sum[i] / count;
	}
}

void StreamMeanStd::std(float *s) {
	for (short i = 0; i < dimension; i++) {
		s[i] = sqrt((max(0.0, ssum[i] - count * pow(sum[i] / count, 2))) / count);
	}
}
