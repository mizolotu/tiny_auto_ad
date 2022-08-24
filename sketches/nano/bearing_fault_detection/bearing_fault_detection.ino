#include <fix_fft_32k.h>
#include <math.h>
#include <Arduino_LSM9DS1.h>

#include "DynamicDimensionQueue.h"
#include "Dense.h"
#include "Svdd.h"
#include "utils.h"

#define XYZ_DIM              3     // xyz dimension for accelerometer data
#define XYZ_Q_SIZE        1024     // queue size for accelerometer data
#define XYZ_SCALE         8192     // xyz scale 32767 / 4g (not sure if this is correct though)
#define FFT_SIZE             1     // number of buffers per fft
#define FFT_N                5     // fft n
#define FFT_STEP             8     // fft step
#define FFT_FEATURES         4     // fft features
#define FFT_Q_SIZE          32     // fft queue size
#define BASELINE_STEPS    1000     // number of iteration to define the baseline
#define BASELINE_ALPHA       3     // baseline number of stds
#define INFERENCE_DELAY     10     // delay during the infrence and training

#define N_LAYERS             2     // number of layers 
#define LAYER_1_UNITS       32     // the 1st layer units
#define LAYER_2_UNITS       16     // the 2nd layer units
#define LEARNING_RATE    0.001     // learning rate
#define TRAIN_COUNT_MAX   1000     // total number of voice instance

#define DEBUG            false     // debug

unsigned int stage = 0;

// Stages:
// 0 - calculating C for deep SVDD
// 1 - training deep SVDD
// 2 - inferencing

unsigned int fft_count = 0;
unsigned int sample_count = 0;
unsigned int total_count = 0;
unsigned int window_count = 0;

DynamicDimensionQueue xyz_q(XYZ_Q_SIZE, XYZ_DIM);
DynamicDimensionQueue fft_q(FFT_Q_SIZE, FFT_FEATURES * XYZ_DIM);

float x[XYZ_DIM];
float x_mean[XYZ_DIM];
float x_std[XYZ_DIM];
float x_std_thr[XYZ_DIM];

bool is_active;

short re[XYZ_Q_SIZE];
short im[XYZ_Q_SIZE];
float freq[FFT_FEATURES * XYZ_DIM];

int idx;
int window_start = 0;

Dense layers[2] = {
  Dense(FFT_FEATURES * XYZ_DIM, LAYER_1_UNITS, &relu, &d_relu),
  Dense(LAYER_1_UNITS, LAYER_2_UNITS, &sigmoid, &d_sigmoid)
};

Svdd model = Svdd(N_LAYERS, layers, LEARNING_RATE);

void setup() {
  
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("Recoding the baseline... THE MOTOR SHOULD BE OFF!");
  
  for (unsigned short i=0; i<XYZ_Q_SIZE; i++) {
  
    // get new xyz data point
  
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x[0], x[1], x[2]);
    }

    // scale the point
    
    for (unsigned short j=0; j<XYZ_DIM; j++) {
      x[j] *= XYZ_SCALE;
    }

    // add the new point to the baseline queue
       
    xyz_q.enqueue(x);

    total_count += 1;

    delay(INFERENCE_DELAY);    
    
  }

  // Save xyz std

  xyz_q.std(x_std_thr);
  
  Serial.println("Training the model... DO NOT MOVE THE BOARD!");

}


void loop() {

  // calculate mean

  xyz_q.mean(x_mean);
  
  // get new xyz data point
  
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x[0], x[1], x[2]);
  }

  // scale the point

  for (unsigned short i=0; i<XYZ_DIM; i++) {
    x[i] *= XYZ_SCALE;
  }

  // enque the point

  xyz_q.enqueue(x);
 
  // check if there is activity

  is_active = false;

  for (unsigned short i=0; i<XYZ_DIM; i++) {
    if (abs(x[i] - x_mean[i]) > BASELINE_ALPHA * x_std_thr[i]) {
      is_active = true;
      break;
    }      
  }

  if (is_active) {
    window_count = FFT_FEATURES;
  }

  if (DEBUG) {
    Serial.print(fft_count);
    Serial.print(", ");
    Serial.print(window_count);
    Serial.println();
  }

  // check that there is enough samples to compute fft

  if (fft_count >= FFT_STEP) {

    // check that more fft features are required

    if (window_count > 0) {

      for (unsigned short j = 0; j < XYZ_DIM; j++) {

        for (unsigned short i = 0; i < FFT_STEP; i++) {
          xyz_q.get(xyz_q.size() - FFT_STEP + i, x);
          re[i] = x[j];
          im[i] = 0;
        }
          
        fix_fft(re, im, FFT_N, 0);

        idx = min(FFT_FEATURES - 1, window_start);
          
        if (window_start > FFT_FEATURES - 1) {
          for (unsigned short i = 0; i < FFT_FEATURES - 1; i++) {
            freq[i * XYZ_DIM + j] = freq[(i + 1) * XYZ_DIM + j];
          }
        } 
          
        freq[idx * XYZ_DIM + j] = sqrt(re[0] * re[0] + im[0] * im[0]) / 2; // only the first frequency is used
          
      }

      if (window_start >= FFT_FEATURES - 1) {
          
        fft_q.enqueue(freq);
     
        if (DEBUG) {
          Serial.println(fft_q.size());
          Serial.println(window_start);
          Serial.println("");
        }

        sample_count += 1;

        if (stage == 0) {

          // updating c
          
          model.forward(freq);
          
        } else if (stage == 1) {

          // updating weights and biases
          
          model.forward(freq);

          //if (DEBUG) {
          Serial.print("Loss: ");
          Serial.print(model.get_score());
          Serial.println("");
          //}
          
          model.backward(freq);
          
        } else {

          // calculating score
          
          model.forward(freq);

          Serial.print("Score: ");
          Serial.print(model.get_score());
          Serial.println("");
          
        }
          
        if (DEBUG) {
          for (unsigned short j = 0; j < FFT_FEATURES * XYZ_DIM; j++) {
            Serial.print(freq[j]);
            Serial.print(",");
          }
          Serial.println("");
          Serial.println("");
        }
          
      }

      window_count -= 1;
      window_start += 1; 

    } else {
      window_start = 0;
    }
      
    fft_count = 0;

  }

  if (stage == 0 && fft_q.isFull()) {

    model.fix_c();    
      
    Serial.println("SVDD center has been found!");
    for (unsigned short j = 0; j < LAYER_2_UNITS; j++) {
      Serial.print(model.get_c(j));
      Serial.print(",");
    }
    Serial.println("");
    Serial.println("");

    sample_count = 0;
    stage = 1;
    
  } else if (stage == 1 && sample_count >= TRAIN_COUNT_MAX) {

    sample_count = 0;
    stage = 2;
    
  }

    //

    /*

    x_mean += x[0];
    y_mean += x[1];
    z_mean += x[2];
    x_std += pow(x, 2);
    y_std += pow(y, 2);
    z_std += pow(z, 2);

    if (sample_count >= BASELINE_STEPS) {

      x_mean = x_mean / BASELINE_STEPS;
      y_mean = y_mean / BASELINE_STEPS;
      z_mean = z_mean / BASELINE_STEPS;

      x_std = sqrt((max(0, x_std - BASELINE_STEPS * pow(x_mean, 2)))/BASELINE_STEPS);
      y_std = sqrt((max(0, y_std - BASELINE_STEPS * pow(y_mean, 2)))/BASELINE_STEPS);
      z_std = sqrt((max(0, z_std - BASELINE_STEPS * pow(z_mean, 2)))/BASELINE_STEPS);

      Serial.println(x_mean, 16);
      Serial.println(y_mean, 16);
      Serial.println(z_mean, 16);
      
      Serial.println(x_std, 16);
      Serial.println(y_std, 16);
      Serial.println(z_std, 16);     

      for (unsigned short i = 0; i < XYZ_BUFFER_SIZE; i++) {
        if (IMU.accelerationAvailable()) {
          IMU.readAcceleration(x, y, z);
        }
        xyz[i][0] = round((x - x_mean) * XYZ_SCALE);
        xyz[i][1] = round((y - y_mean) * XYZ_SCALE);
        xyz[i][2] = round((z - z_mean) * XYZ_SCALE);
        delay(DELAY);
      }

      for (unsigned short j = 0; j < 3; j++) {   
        for (unsigned short i = 0; i < XYZ_BUFFER_SIZE; i++) {
          re[i] = xyz[i][j];
          im[i] = 0;
        }

        fix_fft(re, im, FFT_N, 0);
      
        for (unsigned short i = 0; i < FFT_FEATURES; i++) {
          freq[i * 3 + j] = (int)(sqrt(re[0] * re[0] + im[0] * im[0]) / 2); // only the first frequency is used
        }

      }

      stage = 1;

    }

    */
    
    /*

    for (unsigned short i = 0; i < XYZ_BUFFER_SIZE - 1; i++) {
      for (unsigned short j = 0; j < 3; j++) {
        xyz[i][j] = xyz[i + 1][j];
      }
    }

    if ((abs(x - x_mean) > BASELINE_ALPHA * x_std) || (abs(y - y_mean) > BASELINE_ALPHA * y_std) || (abs(z - z_mean) > BASELINE_ALPHA * z_std)) {
      window_count = FFT_FEATURES;      
    }

    xyz[BUFFER_SIZE - 1][0] = int(x);
    xyz[BUFFER_SIZE - 1][1] = int(y);
    xyz[BUFFER_SIZE - 1][2] = int(z);
    

    if (buffer_count >= FFT_STEP) {

      // FFT

      if (window_count > 0) {

        for (unsigned short j = 0; j < 3; j++) {

          // copy previous samples

          for (unsigned short i = 0; i < FFT_FEATURES - 1; i++) {
            freq[i * 3 + j] = freq[(i + 1) * 3 + j];  
          }
       
          for (unsigned short i = 0; i < XYZ_BUFFER_SIZE; i++) {
            re[i] = xyz[i][j];
            im[i] = 0;
          }

          fix_fft(re, im, FFT_N, 0);
      
          freq[(FFT_FEATURES - 1) * 3  + j] = (int)(sqrt(re[0] * re[0] + im[0] * im[0]) / 2); // only the first frequency is used

        }

        for (unsigned short i = 0; i < FFT_FEATURES; i++) {
          for (unsigned short j = 0; j < 3; j++) {
            Serial.print(freq[i * 3 + j]);
            Serial.print(",");
          }
          Serial.println("");
        }
    
        Serial.println("");
    
        window_count -= 1;    
    
      }
  
      //while (1);
      buffer_count = 0;

    } 

    */
    
  fft_count += 1;
  
  total_count += 1;

  if (DEBUG) {
    Serial.println(total_count);
  }
  
  delay(INFERENCE_DELAY);

}
