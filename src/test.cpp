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
#include "utils.h"

#include <math.h>
#include <unistd.h>
#include "Svdd.h"

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

void testSvd() {

	float x[32][3];
	for (short i=0; i<32; i++) {
		for (short j=0; j<3; j++) {
			x[i][j] = (random() % 1000);
		}
	}
	float y0[4], y1[2], z[2];
	float* y;

	Dense dense0 = Dense(3, 4, &relu, &d_relu);

	Dense dense1 = Dense(4, 2, &sigmoid, &d_sigmoid);

	dense0.forward(x[0]);

	y = dense0.get_outputs();
	for (short k=0; k<4; k++) {
		y0[k] = *(y + k);
	}

	cout << "Dense 0:" << endl;
	for (short k=0; k<4; k++) {
		cout << y0[k] << ", ";
	}
	cout << endl;

	dense1.forward(y0);
	y = dense1.get_outputs();
	for (short k=0; k<4; k++) {
		y1[k] = *(y + k);
	}

	cout << "Dense 1:" << endl;
	for (short k=0; k<2; k++) {
		cout << y1[k] << ", ";
	}
	cout << endl;

	Dense layers[2] = {
		Dense(3, 4, &relu, &d_relu),
		Dense(4, 2, &sigmoid, &d_sigmoid)
	};

	Svdd svd = Svdd(2, layers, 0.01);

	for (int iter=0; iter<32; iter++) {
		svd.forward(x[iter]);
	}

	svd.fix_c();

	float loss;
	for(int epoch=0; epoch < 10000; epoch ++) {

		loss = 0;

		for (int iter=0; iter<32; iter++) {

			svd.forward(x[iter]);

			loss += 0.5 * svd.get_score();

			svd.backward(x[iter]);

		}

		cout << "After epoch " << epoch << " loss = " << loss / 32 << endl;

		usleep(1000);

	}

}

int main() {

	//testStaticDimensionQueue();
	//testDynamicDimensionQueue();
	testSvd();

	return 0;
}
