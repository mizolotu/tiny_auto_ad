/*
 * DynamicDimensionQueue.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <cstdlib>
#include <math.h>
#include <algorithm>

using namespace std;

#include "DynamicDimensionQueue.h"

DynamicDimensionQueue::DynamicDimensionQueue(short cap, short dim) {

	dimension = dim;
	arr = new float[cap * dim];
	for (short i=0; i < cap * dim; i++) {
		arr[i] = 0;
	}
	capacity = cap;
	front = 0;
	rear = -1;
	count = 0;
	sum = new float[dim];
	ssum = new float[dim];
	for (short i=0; i < dim; i++) {
		sum[i] = 0;
		ssum[i] = 0;
	}

}

DynamicDimensionQueue::~DynamicDimensionQueue() {
	delete[] arr;
	delete[] sum;
	delete[] ssum;
}

void DynamicDimensionQueue::dequeue() {

	for (short i=0; i < dimension; i++) {
	 	sum[i] -= arr[front * dimension + i];
	 	ssum[i] -= pow(arr[front * dimension + i], 2);
	}

	front = (front + 1) % capacity;
    count--;

}

void DynamicDimensionQueue::enqueue(float *x) {

    if (isFull()) {
        dequeue();
    }

    rear = (rear + 1) % capacity;

    for (short i=0; i < dimension; i++) {
    	arr[rear * dimension + i] = x[i];
    	sum[i] += x[i];
    	ssum[i] += pow(x[i], 2);
    }

    count++;
}

short DynamicDimensionQueue::size() {
    return count;
}

bool DynamicDimensionQueue::isEmpty() {
    return (size() == 0);
}

bool DynamicDimensionQueue::isFull() {
    return (size() == capacity);
}

void DynamicDimensionQueue::xmax(float* m) {
	for (short i = 0; i < dimension; i++) {
		m[i] = -99999999.0;
		for (short j = 0; j < count; j++) {
			if (arr[j * dimension + i] > m[i]) {
				m[i] = arr[j * dimension + i];
			}
		}
	}
}

void DynamicDimensionQueue::mean(float* m) {
	for (short i = 0; i < dimension; i++) {
		m[i] = sum[i] / count;
	}
}

void DynamicDimensionQueue::std(float *s) {
	for (short i = 0; i < dimension; i++) {
		s[i] = sqrt((max(0.0, ssum[i] - count * pow(sum[i] / count, 2))) / count);
	}
}

float* DynamicDimensionQueue::get(short i) {
	float *x = new float[dimension];
	for (short j = 0; j < dimension; j++) {
		x[j] = arr[((i + front) % capacity) * dimension + j];
	}
	return x;
}

void DynamicDimensionQueue::get(short i, float* x) {
	for (short j = 0; j < dimension; j++) {
		x[j] = arr[((i + front) % capacity) * dimension + j];
	}
}

