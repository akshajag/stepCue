/*
 * StepCue - ESP32 waist hub firmware
 *
 * Hardware (all wired from the waist hub):
 *   Ankle node : MPU6050 (I2C) + high-brightness cue LED
 *   Wrist      : DC vibration motor through a motor driver (PWM)
 *
 * LOCKED CONTRACT (do not change):
 *   SoftAP  SSID "StepCue_AP"  PASS "stepcue123"   ESP32 = 192.168.4.1
 *   Sensor  UDP -> 192.168.4.2:5005 at 50 Hz, one CSV line per packet:
 *             ts_ms,ax,ay,az,gx,gy,gz\n      (accel in g, gyro in deg/s)
 *   Cues    UDP <- port 5006:
 *             CUE:<bpm>:<haptic_0_or_1>      start / update rhythmic cue
 *             STOP                           stop cueing immediately
 *
 * Behaviour:
 *   While cueing, every beat flashes the ankle LED and (if haptic=1) drives
 *   the wrist motor for a 150 ms pulse. The laptop re-sends CUE about once a
 *   second; if no CUE arrives for 5 s the cue stops on its own (failsafe for a
 *   crashed laptop or dropped Wi-Fi). A repeated CUE keeps the beat phase.
 *
 * NOT YET RUN ON HARDWARE. Check the pin numbers and motor-driver wiring
 * against your build before powering the motor.
 */
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// ------------------------------------------------------------- pins -------
const int PIN_SDA       = 21;
const int PIN_SCL       = 22;
const int PIN_LED       = 27;   // ankle cue LED (through a transistor if >20 mA)
const int PIN_MOTOR_PWM = 25;   // motor driver IN1 / PWM input
const int PIN_MOTOR_DIR = 26;   // motor driver IN2 (held LOW), set -1 if unused

// ------------------------------------------------------------ network -----
const char* AP_SSID = "StepCue_AP";
const char* AP_PASS = "stepcue123";
const IPAddress AP_IP(192, 168, 4, 1);
const IPAddress AP_GATEWAY(192, 168, 4, 1);
const IPAddress AP_MASK(255, 255, 255, 0);
const IPAddress LAPTOP_IP(192, 168, 4, 2);
const uint16_t SENSOR_PORT = 5005;
const uint16_t CUE_PORT    = 5006;

// ------------------------------------------------------------- timing -----
const uint32_t SAMPLE_PERIOD_US = 20000;   // 50 Hz
const uint32_t PULSE_MS         = 150;     // LED flash + wrist pulse length
const uint32_t CUE_FAILSAFE_MS  = 5000;
const int      BPM_MIN = 30, BPM_MAX = 200;

// -------------------------------------------------------------- motor -----
const int MOTOR_PWM_FREQ = 20000;          // above hearing range
const int MOTOR_PWM_BITS = 8;
const int MOTOR_DUTY     = 255;            // 0-255, lower it if the buzz is too strong
const int MOTOR_CHANNEL  = 0;              // only used on ESP32 core 2.x

// ------------------------------------------------------------ MPU6050 -----
const uint8_t MPU_ADDR = 0x68;
const float ACCEL_LSB_PER_G   = 4096.0f;   // +/-8 g range
const float GYRO_LSB_PER_DPS  = 32.8f;     // +/-1000 deg/s range

WiFiUDP udpSensor;
WiFiUDP udpCue;

bool     mpuOk        = false;
bool     cueActive    = false;
int      cueBpm       = 100;
bool     cueHaptic    = false;
uint32_t beatPeriodMs = 600;
uint32_t nextBeatMs   = 0;
uint32_t pulseEndMs   = 0;
bool     pulseOn      = false;
uint32_t lastCueRxMs  = 0;
uint32_t nextSampleUs = 0;

// ---------------------------------------------------------------------------
void motorWrite(int duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PIN_MOTOR_PWM, duty);
#else
  ledcWrite(MOTOR_CHANNEL, duty);
#endif
}

void motorSetup() {
  if (PIN_MOTOR_DIR >= 0) {
    pinMode(PIN_MOTOR_DIR, OUTPUT);
    digitalWrite(PIN_MOTOR_DIR, LOW);
  }
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PIN_MOTOR_PWM, MOTOR_PWM_FREQ, MOTOR_PWM_BITS);
#else
  ledcSetup(MOTOR_CHANNEL, MOTOR_PWM_FREQ, MOTOR_PWM_BITS);
  ledcAttachPin(PIN_MOTOR_PWM, MOTOR_CHANNEL);
#endif
  motorWrite(0);
}

void outputsOff() {
  digitalWrite(PIN_LED, LOW);
  motorWrite(0);
  pulseOn = false;
}

void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool mpuSetup() {
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) return false;
  mpuWrite(0x6B, 0x01);   // wake, clock = gyro X PLL
  delay(50);
  mpuWrite(0x1A, 0x04);   // DLPF ~20 Hz (below the 25 Hz Nyquist limit)
  mpuWrite(0x1B, 0x10);   // gyro  +/-1000 deg/s
  mpuWrite(0x1C, 0x10);   // accel +/-8 g
  return true;
}

bool mpuRead(float& ax, float& ay, float& az, float& gx, float& gy, float& gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  uint8_t b[14];
  for (int i = 0; i < 14; i++) b[i] = Wire.read();
  int16_t rax = (b[0] << 8) | b[1],  ray = (b[2] << 8) | b[3],   raz = (b[4] << 8) | b[5];
  int16_t rgx = (b[8] << 8) | b[9],  rgy = (b[10] << 8) | b[11], rgz = (b[12] << 8) | b[13];
  ax = rax / ACCEL_LSB_PER_G;  ay = ray / ACCEL_LSB_PER_G;  az = raz / ACCEL_LSB_PER_G;
  gx = rgx / GYRO_LSB_PER_DPS; gy = rgy / GYRO_LSB_PER_DPS; gz = rgz / GYRO_LSB_PER_DPS;
  return true;
}

// ------------------------------------------------------------ cue logic ----
void startOrUpdateCue(int bpm, bool haptic) {
  bpm = constrain(bpm, BPM_MIN, BPM_MAX);
  uint32_t now = millis();
  bool wasActive = cueActive;
  cueBpm = bpm;
  cueHaptic = haptic;
  beatPeriodMs = 60000UL / (uint32_t)bpm;
  lastCueRxMs = now;
  cueActive = true;
  if (!wasActive) nextBeatMs = now;         // first beat right away
  if (!haptic && pulseOn) motorWrite(0);    // haptics switched off mid-pulse
}

void stopCue() {
  cueActive = false;
  outputsOff();
}

void handleCommand(char* msg) {
  if (strncmp(msg, "STOP", 4) == 0) {
    stopCue();
    return;
  }
  int bpm = 0, haptic = 0;
  if (sscanf(msg, "CUE:%d:%d", &bpm, &haptic) == 2) {
    startOrUpdateCue(bpm, haptic != 0);
  }
}

void pollCommands() {
  int len = udpCue.parsePacket();
  while (len > 0) {
    char buf[48];
    int n = udpCue.read(buf, sizeof(buf) - 1);
    if (n > 0) {
      buf[n] = '\0';
      handleCommand(buf);
    }
    len = udpCue.parsePacket();
  }
}

void serviceCue() {
  uint32_t now = millis();
  if (cueActive && (now - lastCueRxMs) > CUE_FAILSAFE_MS) stopCue();
  if (pulseOn && (int32_t)(now - pulseEndMs) >= 0) outputsOff();
  if (cueActive && (int32_t)(now - nextBeatMs) >= 0) {
    digitalWrite(PIN_LED, HIGH);
    if (cueHaptic) motorWrite(MOTOR_DUTY);
    pulseOn = true;
    pulseEndMs = now + PULSE_MS;
    nextBeatMs += beatPeriodMs;
    if ((int32_t)(now - nextBeatMs) >= 0) nextBeatMs = now + beatPeriodMs;  // fell behind
  }
}

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  motorSetup();

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  mpuOk = mpuSetup();
  Serial.println(mpuOk ? "MPU6050 ready" : "MPU6050 NOT FOUND - check ankle wiring");

  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(AP_IP, AP_GATEWAY, AP_MASK);
  WiFi.softAP(AP_SSID, AP_PASS);
  WiFi.setSleep(false);
  Serial.print("SoftAP up, IP ");
  Serial.println(WiFi.softAPIP());

  udpSensor.begin(SENSOR_PORT);
  udpCue.begin(CUE_PORT);
  nextSampleUs = micros();
}

void loop() {
  pollCommands();
  serviceCue();

  if ((int32_t)(micros() - nextSampleUs) >= 0) {
    nextSampleUs += SAMPLE_PERIOD_US;
    if ((int32_t)(micros() - nextSampleUs) > (int32_t)(5 * SAMPLE_PERIOD_US)) {
      nextSampleUs = micros() + SAMPLE_PERIOD_US;       // resync after a stall
    }
    if (!mpuOk) {
      static uint32_t lastRetry = 0;
      if (millis() - lastRetry > 1000) {                // IMU missing: retry once a second
        lastRetry = millis();
        mpuOk = mpuSetup();
      }
      return;
    }
    float ax, ay, az, gx, gy, gz;
    if (mpuRead(ax, ay, az, gx, gy, gz)) {
      char pkt[96];
      int n = snprintf(pkt, sizeof(pkt), "%lu,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n",
                       (unsigned long)millis(), ax, ay, az, gx, gy, gz);
      udpSensor.beginPacket(LAPTOP_IP, SENSOR_PORT);
      udpSensor.write((const uint8_t*)pkt, n);
      udpSensor.endPacket();
    } else {
      mpuOk = false;
    }
  }
}
