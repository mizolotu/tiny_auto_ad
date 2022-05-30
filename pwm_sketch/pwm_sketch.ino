#define RAMP_UP_TIME 3000  // in ms
#define ON_TIME 5000
#define RAMP_DOWN_TIME 3000
#define OFF_TIME 5000

#define TICK 100  // Time between pwm adjustments (ms)
#define UP_INC (255 / (RAMP_UP_TIME / TICK))
#define DOWN_INC (255 / (RAMP_DOWN_TIME / TICK))

#define PWM_PIN 6

int speed = 0;

void rampUp() {
  Serial.println("Ramping up...");
  while (speed < 255) {
    analogWrite(PWM_PIN, speed);
    Serial.println(speed);
    speed += UP_INC;
    delay(TICK);
  }
  speed = 255;
  Serial.println(speed);
}

void rampDown() {
  Serial.println("Ramping down...");
  while (speed > 0) {
    analogWrite(PWM_PIN, speed);
    Serial.println(speed);
    speed -= DOWN_INC;
    delay(TICK);
  }
  speed = 0;
  Serial.println(speed);
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(PWM_PIN, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println("Running cycle:");
  rampUp();
  delay(ON_TIME);
  rampDown();
  delay(OFF_TIME);
  Serial.println("End of cycle");
}
