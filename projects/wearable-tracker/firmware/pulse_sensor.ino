#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  delay(1000);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found. Check wiring.");
    while (1);
  }

  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);

  Serial.println("Sensor ready. Place finger on sensor.");
}

void loop() {
  long irValue = particleSensor.getIR();
  Serial.println(irValue);
  delay(10);
}
