#include <fix_fft_32k.h>
#include <math.h>
#include <Arduino_LSM9DS1.h>


#define BUFFER_SIZE         32     // buffer size for accelerometer data
#define BUFFER_DIM           3     // buffer size for accelerometer data
#define FFT_SIZE             1     // number of buffers per fft
#define FFT_N                5     // fft n
#define FFT_STEP             8     // fft step
#define FFT_FEATURES         4     // fft step
#define XYZ_SCALE         8192     // XYZ scale 32767 / 4g (not sure if this is correct though)
#define BASELINE_STEPS    1000     // number of iteration to define the baseline
#define BASELINE_ALPHA       3     // baseline number of stds
#define INFERENCE_DELAY     10     // delay during the infrence and training
#define BASELINE_DELAY_MP  100     // inference delay multiplier for the baseline recording
#define BATCH_SIZE          32     // batch size

#define SAMPLE_THRESHOLD  1000  // XYZ threshold to trigger sampling
#define FEATURE_SIZE      1     // sampling size of one voice instance
#define TOTAL_COUNT_MAX   20000 // total number of voice instance
#define SCORE_THRESHOLD   0.5   // score threshold 


class Queue {
  float (*arr)[BUFFER_DIM];     // array to store queue elements
  int capacity;                 // maximum capacity of the queue
  int front;                    // front points to the front element in the queue (if any)
  int rear;                     // rear points to the last element in the queue
  int count;                    // current size of the queue
  float arr_sum;
  float arr_ssum;

private:

  void dequeue();

public:

  Queue(short size = BUFFER_SIZE);     // constructor
  ~Queue();                            // destructor

  void enqueue(float x[BUFFER_DIM]);
  int size();
  bool isEmpty();
  bool isFull();
  float* mean();
  
};

Queue::Queue(short size) {
  arr = new float[size][BUFFER_DIM];
  capacity = size;
  front = 0;
  rear = -1;
  count = 0;
}

Queue::~Queue() {
  delete[] arr;
}

void Queue::dequeue() {
  front = (front + 1) % capacity;
  count--;
}

void Queue::enqueue(float x[BUFFER_DIM]) { 
    
  if (isFull()) {
    dequeue();
  }

  rear = (rear + 1) % capacity;

  for (short i=0; i < BUFFER_DIM; i++) {
    arr[rear][i] = x[i];
  }

  count++;
}

int Queue::size() {
  return count;
}

bool Queue::isEmpty() {
  return (size() == 0);
}

bool Queue::isFull() {
  return (size() == capacity);
}

float* Queue::mean() {
  static float m[BUFFER_DIM];
  for (short j = 0; j < BUFFER_DIM; j++) {
    for (short i = 0; i < capacity; i++) {
      m[j] += arr[i][j];
    }
    m[j] /= capacity;
  }
  return m;
}


unsigned int stage = 0;

// Stages:
// 0 - calculating C
// 1 - training deep SVDD
// 2 - inference

unsigned int buffer_count = 0;
unsigned int sample_count = 0;
unsigned int total_count = 0;
unsigned int window_count = 0;

float x[BUFFER_DIM];
float x_min, y_min, z_min;
float x_max, y_max, z_max;
float x_mean, y_mean, z_mean;
float x_std, y_std, z_std;

short re[XYZ_BUFFER_SIZE];
short im[XYZ_BUFFER_SIZE];
float freq[FFT_FEATURES * 3];
float batch[BATCH_SIZE][FFT_FEATURES * 3];

Queue base_q(BUFFER_SIZE);
Queue xyz_q(BUFFER_SIZE);
Queue fft_q(BUFFER_SIZE);

void setup() {
  
  Serial.begin(115200);
  while (!Serial);
  
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("Recoding baseline!");

  for (unsigned short i=0; i<BUFFER_SIZE; i++) {
  
    // get new xyz data point
  
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x[0], x[1], x[2]);
    }

    // scale the point
    
    for (unsigned short j=0; j<BUFFER_DIM; j++) {
      x[j] *= XYZ_SCALE;
    }

    // add the new point to the baseline queue
       
    base_q.enqueue(x);

    delay(BASELINE_DELAY_MP * INFERENCE_DELAY);
    
  }
}


void loop() {

  
  // get new xyz data point
  
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x[0], x[1], x[2]);
  }

  for (unsigned short i=0; i<BUFFER_DIM; i++) {
    x[i] *= XYZ_SCALE;
  }

  if (sample_count % BUFFER_SIZE == 0) {
    base_q.enqueue(x);  
  }
   
  if (stage == 0) {

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

    
  } else if (stage == 1) {

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
  }

  buffer_count += 1;
  sample_count += 1;
  
  delay(DELAY);
    
}
