/*
 * utils.cpp
 *
 *  Created on: Aug 23, 2022
 *      Author: mizolotu
 */

#include <cstdlib>
#include <math.h>
#include <algorithm>

using namespace std;

#include "utils.h"

float relu(float x) {
    return fmaxf(0.0f, x);
}

float d_relu(float x) {
	float d;
	if (x > 0) {
		d = 1;
	} else {
		d = 0;
	}
    return d;
}


