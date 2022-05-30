#include <Arduino_LSM9DS1.h>

#define DELAY 10


float x, y, z;

void setup() {
  
  Serial.begin(115200);
  while (!Serial);
  
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

}

void loop(void) {
  
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);
  }
  
  Serial.print('<');
  Serial.print(x);  
  Serial.print(',');
  Serial.print(y);  
  Serial.print(',');
  Serial.print(z);
  Serial.print('>');
  Serial.println();
  
  delay(DELAY);
  
}
