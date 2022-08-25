/*
 * DynamicDimensionQueue.h
 *
 *  Created on: Aug 11, 2022
 *      Author: mizolotu
 */

#ifndef DYNAMICDIMENSIONQUEUE_H_
#define DYNAMICDIMENSIONQUEUE_H_

class DynamicDimensionQueue {

	float *arr;          // array to store queue elements
	short dimension;     // dimension of the queue
	short capacity;      // maximum capacity of the queue
	short front;         // front points to the front element in the queue (if any)
	short rear;          // rear points to the last element in the queue
	short count;         // current size of the queue
	float *sum;          // sum of elements
	float *ssum;         // sum of element squares

	private:

	public:

		DynamicDimensionQueue(short cap, short dim);
		~DynamicDimensionQueue();

		void enqueue(float *x);       // add an element to the queue
		void dequeue();               // remove the first element
		short size();                 // get the queue size
		bool isEmpty();               // check whether the queue is empty
		bool isFull();                // check whether the queue is full
		void xmax(float* m);            // calculate the queue max
		void mean(float* m);          // calculate the queue mean
		void std(float* s);           // calculate the queue std
		float* get(short i);          // get an element of the queue
		void get(short i, float* x);  // get an element of the queue

};

#endif /* DYNAMICDIMENSIONQUEUE_H_ */
