/*
 * DynamicSizeQueue.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <cstdlib>
#include <math.h>
#include <algorithm>

using namespace std;

#include "DynamicDimensionQueue.h"

DynamicSizeQueue::DynamicSizeQueue(short cap, short dim) {

	dimension = dim;
    arr = new float[cap * dim];
    capacity = cap;
    front = 0;
    rear = -1;
    count = 0;
    sum = new float[dim];
    ssum = new float[dim];

}

DynamicSizeQueue::~DynamicSizeQueue() {
    delete[] arr;
    delete[] sum;
    delete[] ssum;
}

void DynamicSizeQueue::dequeue() {

	for (short i=0; i < dimension; i++) {
	 	sum[i] -= arr[front * dimension + i];
	 	ssum[i] -= pow(arr[front * dimension + i], 2);
	}

	front = (front + 1) % capacity;
    count--;

}

void DynamicSizeQueue::enqueue(float *x) {

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

short DynamicSizeQueue::size() {
    return count;
}

bool DynamicSizeQueue::isEmpty() {
    return (size() == 0);
}

bool DynamicSizeQueue::isFull() {
    return (size() == capacity);
}

float* DynamicSizeQueue::mean() {
	float *m = new float[dimension];
	for (short i = 0; i < dimension; i++) {
		m[i] = sum[i] / count;
	}
	return m;
}

float* DynamicSizeQueue::std() {
	float *s = new float[dimension];
	for (short i = 0; i < dimension; i++) {
		s[i] = sqrt((max(0.0, ssum[i] - count * pow(sum[i] / count, 2))) / count);
	}
	return s;
}

float* DynamicSizeQueue::get(short i) {
	float *x = new float[dimension];
	for (short j = 0; j < dimension; j++) {
		x[j] = arr[((i + front) % capacity) * dimension + j];
	}
	return x;
}

