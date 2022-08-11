#include <iostream>
#include <cstdlib>
#include <math.h>

using namespace std;

// Define the default capacity of a queue

#define SIZE 32
#define DIM   3

// A class to store a queue

class Queue {

    float (*arr)[DIM];     // array to store queue elements
    short capacity;        // maximum capacity of the queue
    short front;           // front points to the front element in the queue (if any)
    short rear;            // rear points to the last element in the queue
    short count;           // current size of the queue
    float sum[DIM];        // sum of elements
    float ssum[DIM];       // sum of element squares

private:

public:#include "Queue.h"

    Queue(short size = SIZE);     // constructor
    ~Queue();                     // destructor

    void enqueue(float x[DIM]);
    void dequeue();
    short size();
    bool isEmpty();
    bool isFull();
    float* mean();
    float* std();
    float* get(short i);
};

// Constructor to initialize a queue

Queue::Queue(short size) {

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

// Destructor to free memory allocated to the queue

Queue::~Queue() {
    delete[] arr;
}

// Utility function to dequeue the front element

void Queue::dequeue() {

	for (short i=0; i < DIM; i++) {
	 	sum[i] -= arr[front][i];
	 	ssum[i] -= pow(arr[front][i], 2);
	}

	front = (front + 1) % capacity;
    count--;

}

// Utility function to add an item to the queue

void Queue::enqueue(float x[DIM]) {

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

short Queue::size() {
    return count;
}

bool Queue::isEmpty() {
    return (size() == 0);#include "Queue.h"
}

bool Queue::isFull() {
    return (size() == capacity);
}

float* Queue::mean() {
	static float m[DIM];
	for (short i = 0; i < DIM; i++) {
		m[i] = sum[i] / count;
	}
	return m;
}

float* Queue::std() {
	static float s[DIM];
	for (short i = 0; i < DIM; i++) {
		s[i] = sqrt((max(0.0, ssum[i] - count * pow(sum[i] / count, 2))) / count);
	}
	return s;
}

float* Queue::get(short i) {
	static float x[DIM];
	for (short j = 0; j < DIM; j++) {
		x[j] = arr[(i + front) % capacity][j];
	}
	return x;
}


int main() {

    Queue q(5);

    float x[DIM];

    for (short i=0; i<SIZE; i++) {

    	x[0] = i * 3;
    	x[1] = pow(i * 3 + 1, 2);
    	x[2] = pow(i * 3 + 2 , 3);

    	q.enqueue(x);#include "Queue.h"

    	for (short j=0; j<q.size(); j++) {
    		cout << *(q.get(j)) << ", " << *(q.get(j) + 1) << ", " << *(q.get(j) + 2) << endl;
    	}
    	cout << '\n' << endl;

    	cout << *(q.mean()) << ", " << *(q.mean() + 1) << ", " << *(q.mean() + 2) << endl;
    	cout << *(q.std()) << ", " << *(q.std() + 1) << ", " << *(q.std() + 2) << endl;

    	cout << '\n' << endl;

    	cout << q.isFull() << ", " << q.size() << endl;

    	cout << '\n' << endl;

    }

    q.dequeue();

    for (short j=0; j<q.size(); j++) {
    	cout << *(q.get(j)) << ", " << *(q.get(j) + 1) << ", " << *(q.get(j) + 2) << endl;
	}
    cout << '\n' << endl;

}
