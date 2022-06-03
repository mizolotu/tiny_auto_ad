#include <Arduino_LSM9DS1.h>

#define DELAY 10
#define PLOTTER false
#define PRINT_MEAN false
#define BUFFER_SIZE 1000 

float x, y, z;
float x_sum, y_sum, z_sum;
float x_buffer[BUFFER_SIZE], y_buffer[BUFFER_SIZE], z_buffer[BUFFER_SIZE];

void setup() {
  
  Serial.begin(115200);
  while (!Serial);
  
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  for (unsigned short i = 0; i < BUFFER_SIZE; i++) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(x, y, z);
    }
    x_buffer[i] = x;
    y_buffer[i] = y;
    z_buffer[i] = z;
    x_sum += x;
    y_sum += y;
    z_sum += z;
  }  

}

void loop(void) {
  
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);
  }

  x_sum -= x_buffer[0];
  y_sum -= y_buffer[0];
  z_sum -= z_buffer[0];

  for (unsigned short i = 0; i < BUFFER_SIZE - 1; i++) {    
    x_buffer[i] = x_buffer[i + 1];
    y_buffer[i] = y_buffer[i + 1];
    z_buffer[i] = z_buffer[i + 1];
  }

  x_buffer[BUFFER_SIZE-1] = x;
  y_buffer[BUFFER_SIZE-1] = y;
  z_buffer[BUFFER_SIZE-1] = z;

  x_sum += x;
  y_sum += y;
  z_sum += z;
  
  if (!PLOTTER) Serial.print('<');
  (!PRINT_MEAN) ? Serial.print(x_sum / BUFFER_SIZE, 16) : Serial.print(x, 16);
  (PLOTTER) ? Serial.print(' ') : Serial.print(',');
  (!PRINT_MEAN) ? Serial.print(y_sum / BUFFER_SIZE, 16) : Serial.print(y, 16);
  (PLOTTER) ? Serial.print(' ') : Serial.print(',');  
  (!PRINT_MEAN) ? Serial.print(z_sum / BUFFER_SIZE, 16) : Serial.print(z, 16);
  if (!PLOTTER) Serial.print('>');
  Serial.println();
  
  delay(DELAY);
  
}
