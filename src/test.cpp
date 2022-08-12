//============================================================================
// Name        : test.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "Dense.h"
#include "DynamicDimensionQueue.h"
#include "StaticDimensionQueue.h"
#include <math.h>
#include <unistd.h>

#define SIZE 5
#define DIM  3

void testStaticDimensionQueue() {

	cout << "Testing queue" << endl << endl;

	StaticDimensionQueue q(5);

	float x[DIM];

	for (short i=0; i<SIZE; i++) {

		x[0] = i * 3;
		x[1] = pow(i * 3 + 1, 2);
		x[2] = pow(i * 3 + 2 , 3);

		q.enqueue(x);

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

void testDynamicDimensionQueue() {

	cout << "Testing queue" << endl << endl;

	DynamicDimensionQueue q(SIZE, DIM);

	float x[DIM];
	float y[DIM];
	float m[DIM];
	float s[DIM];

	for (short i=0; i<10; i++) {

		x[0] = i * 3;
		x[1] = pow(i * 3 + 1, 2);
		x[2] = pow(i * 3 + 2 , 3);

		q.enqueue(x);

		for (short j=0; j<q.size(); j++) {
			q.get(j, y);
			for (short k=0; k<DIM; k++) {
				cout << y[k] << ", ";
			}
			cout << endl;
	    }

		cout << '\n' << endl;

		q.mean(m);
		for (short k=0; k<DIM; k++) {
			cout << m[k] << ", ";
		}
		cout << endl;

		q.std(s);
		for (short k=0; k<DIM; k++) {
			cout << s[k] << ", ";
		}
		cout << endl;

		cout << '\n' << endl;

	   	cout << q.isFull() << ", " << q.size() << endl;

	   	cout << '\n' << endl;

		usleep(10000);

    }

	q.dequeue();

	for (short j=0; j<q.size(); j++) {
		q.get(j, y);
		for (short k=0; k<DIM; k++) {
			cout << y[k] << ", ";
		}
		cout << endl;
	}
	cout << '\n' << endl;

}

void testDense() {
	cout << (random() % 1000) / 1000.0 << endl << endl;
}

int main() {

	//testStaticDimensionQueue();
	testDynamicDimensionQueue();
	//testDense();

	return 0;
}
