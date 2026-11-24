// =============================================================================
// AirBeats DAQ Firmware
// Arduino Mega 2560
//
// Acquires:
//   - Pressure sensor     (A0)
//   - Flex sensors 1-4    (A1-A4)
//   - IMU 6-axis          (I2C, address 0x68, AlashMPU library)
//
// Serial packet format (one line per sample):
//   timestamp_ms,pressure,flex1,flex2,flex3,flex4,ax,ay,az,gx,gy,gz
//
// Baud rate  : 115200
// Sample rate: 50 Hz (20 ms interval)
// =============================================================================

#include <Wire.h>
#include <AlashMPU.h>

// -----------------------------------------------------------------------------
// Pin configuration
// -----------------------------------------------------------------------------
static const int PIN_PRESSURE = A0;
static const int PIN_FLEX1    = A1;
static const int PIN_FLEX2    = A2;
static const int PIN_FLEX3    = A3;
static const int PIN_FLEX4    = A4;

// -----------------------------------------------------------------------------
// Timing configuration
// -----------------------------------------------------------------------------
static const unsigned long SAMPLE_INTERVAL_MS = 20;   // 50 Hz
static const unsigned long SERIAL_BAUD        = 115200;

// -----------------------------------------------------------------------------
// IMU instance
// -----------------------------------------------------------------------------
AlashMPU mpu;

// -----------------------------------------------------------------------------
// Timing state
// -----------------------------------------------------------------------------
unsigned long lastSampleTime = 0;

// -----------------------------------------------------------------------------
// Setup
// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);

  // Allow serial monitor to connect before we send anything
  delay(500);

  // Initialise I2C bus and IMU
  Wire.begin();
  
  if (!mpu.begin()) {
    // Report failure and halt. Python will see only error lines, never valid
    // packets, making the failure immediately visible.
    Serial.println("ERROR:IMU_INIT_FAILED");
    while (true) {
      delay(1000);
      Serial.println("ERROR:IMU_INIT_FAILED");
    }
  }

  // Calibrate the IMU offsets. Keep the glove completely still during setup.
  // The AlashMPU calibrate() function averages several readings to estimate
  // gyroscope bias and accelerometer offset.
  mpu.calibrate();

  // Tell the Python serial reader that the firmware is running and ready.
  Serial.println("READY");
}

// -----------------------------------------------------------------------------
// Main loop
// Non-blocking: checks elapsed time instead of calling delay().
// delay() would stall the CPU and prevent anything else from running.
// -----------------------------------------------------------------------------
void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;
    sendPacket(now);
  }
}

// -----------------------------------------------------------------------------
// sendPacket
// Reads all sensors and transmits one CSV line to the serial port.
//
// Packet format:
//   timestamp_ms,pressure,flex1,flex2,flex3,flex4,ax,ay,az,gx,gy,gz
//
// Analog values are raw 10-bit ADC integers (0-1023).
// Accelerometer values are in g (gravitational units).
// Gyroscope values are in degrees per second.
// println() appends \r\n which the Python parser uses as the packet terminator.
// -----------------------------------------------------------------------------
void sendPacket(unsigned long timestamp_ms) {

  // Analog reads (10-bit ADC, 0-1023)
  int pressure = analogRead(PIN_PRESSURE);
  int flex1    = analogRead(PIN_FLEX1);
  int flex2    = analogRead(PIN_FLEX2);
  int flex3    = analogRead(PIN_FLEX3);
  int flex4    = analogRead(PIN_FLEX4);

  // IMU reads via AlashMPU
  // accelX/Y/Z : gravitational units (g)
  // gyroX/Y/Z  : degrees per second (deg/s)
  float ax = mpu.accelX();
  float ay = mpu.accelY();
  float az = mpu.accelZ();
  float gx = mpu.gyroX();
  float gy = mpu.gyroY();
  float gz = mpu.gyroZ();

  // Transmit packet as comma-separated values, terminated by newline
  Serial.print(timestamp_ms);  Serial.print(',');
  Serial.print(pressure);       Serial.print(',');
  Serial.print(flex1);          Serial.print(',');
  Serial.print(flex2);          Serial.print(',');
  Serial.print(flex3);          Serial.print(',');
  Serial.print(flex4);          Serial.print(',');
  Serial.print(ax, 4);          Serial.print(',');
  Serial.print(ay, 4);          Serial.print(',');
  Serial.print(az, 4);          Serial.print(',');
  Serial.print(gx, 4);          Serial.print(',');
  Serial.print(gy, 4);          Serial.print(',');
  Serial.println(gz, 4);        // println adds \r\n — packet terminator
}
