/*
 * Queue.cpp
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include "StaticDimensionQueue.h"

StaticDimensionQueue::StaticDimensionQueue(short size) {

    arr = new float[size][3];
    capacity = size;
    front = 0;
    rear = -1;
    count = 0;
    for (short i=0; i < DIM; i++) {
    	sum[i] = 0;
    	ssum[i] = 0;
    }

}

StaticDimensionQueue::~StaticDimensionQueue() {
    delete[] arr;
}

void StaticDimensionQueue::dequeue() {

	for (short i=0; i < DIM; i++) {
	 	sum[i] -= arr[front][i];
	 	ssum[i] -= pow(arr[front][i], 2);
	}

	front = (front + 1) % capacity;
    count--;

}

void StaticDimensionQueue::enqueue(float x[DIM]) {

    if (isFull()) {
        dequeue();
    }

    rear = (rear + 1) % capacity;

    for (short i=0; i < DIM; i++) {
    	arr[rear][i] = x[i];
    	sum[i] += x[i];
    	ssum[i] += pow(x[i], 2);
    }

    count++;
}

short StaticDimensionQueue::size() {
    return count;
}

bool StaticDimensionQueue::isEmpty() {
    return (size() == 0);
}

bool StaticDimensionQueue::isFull() {
    return (size() == capacity);
}

float* StaticDimensionQueue::mean() {
	static float m[DIM];
	for (short i = 0; i < DIM; i++) {
		m[i] = sum[i] / count;
	}
	return m;
}

float* StaticDimensionQueue::std() {
	static float s[DIM];
	for (short i = 0; i < DIM; i++) {
		s[i] = sqrt((max(0.0, ssum[i] - count * pow(sum[i] / count, 2))) / count);
	}
	return s;
}

float* StaticDimensionQueue::get(short i) {
	static float x[DIM];
	for (short j = 0; j < DIM; j++) {
		x[j] = arr[(i + front) % capacity][j];
	}
	return x;
}

