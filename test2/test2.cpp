#include <iostream>
#include <cstdlib>
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
    float sum;             // sum of elements
    float ssum;             // sum of element squares

private:

    void dequeue();

public:

    Queue(short size = SIZE);     // constructor
    ~Queue();                     // destructor

    void enqueue(float x[DIM]);
    short size();
    bool isEmpty();
    bool isFull();
    float* mean();
    float* get(short i);

};

// Constructor to initialize a queue

Queue::Queue(short size) {

    arr = new float[size][3];
    capacity = size;
    front = 0;
    rear = -1;
    count = 0;

}

// Destructor to free memory allocated to the queue

Queue::~Queue() {
    delete[] arr;
}

// Utility function to dequeue the front element

void Queue::dequeue() {
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
    }

    count++;
}

short Queue::size() {
    return count;
}

bool Queue::isEmpty() {
    return (size() == 0);
}

bool Queue::isFull() {
    return (size() == capacity);
}

float* Queue::mean() {
	static float m[DIM];
	for (short j = 0; j < DIM; j++) {
		for (short i = 0; i < capacity; i++) {
	    	m[j] += arr[i][j];
	    }
		m[j] /= capacity;
	}
	return m;
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

    	x[0] = i;
    	x[1] = i + 1;
    	x[2] = i + 2;

    	q.enqueue(x);

    	cout << *(q.get(0)) << endl;

    	//cout << *(q.get(0)) << ", " << *(q.get(1)) << ", " << *(q.get(2)) << ", " << *(q.get(3)) << ", " << *(q.get(4)) << endl;
    	//cout << q.isFull() << ", " << q.size() << ", " <<  *(q.mean()) << endl;
    }

}
