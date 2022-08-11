/*
 * Queue.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#include <cstdlib>
#include <math.h>
#include <algorithm>

using namespace std;

#ifndef QUEUE_H_
#define QUEUE_H_

#define DIM   3

class StaticDimensionQueue {

    float (*arr)[DIM];     // array to store queue elements
    short capacity;        // maximum capacity of the queue
    short front;           // front points to the front element in the queue (if any)
    short rear;            // rear points to the last element in the queue
    short count;           // current size of the queue
    float sum[DIM];        // sum of elements
    float ssum[DIM];       // sum of element squares

	private:

	public:

    	StaticDimensionQueue(short size);
    	~StaticDimensionQueue();

    	void enqueue(float x[DIM]);   // add an element to the queue
    	void dequeue();               // remove the first element
    	short size();                 // get the queue size
    	bool isEmpty();               // check whether the queue is empty
    	bool isFull();                // check whether the queue is full
    	float* mean();                // calculate the queue mean
    	float* std();                 // calculate the queue std
    	float* get(short i);          // get an element of the queue

};

#endif /* QUEUE_H_ */
