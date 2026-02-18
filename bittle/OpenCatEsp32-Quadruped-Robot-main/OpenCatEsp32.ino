/*
  Minimal Bittle RL runtime firmware.

  Goal:
  - Keep only what is required to run trained policy outputs on physical Bittle.
  - Preserve OpenCat servo mapping, direction, and calibration math.

  Hardware target:
  - BiBoard V0_2 + Bittle (8 walking joints only).

  Serial protocol (115200 baud, newline terminated):
  - A d0 d1 d2 d3 d4 d5 d6 d7
      8 target joint angles in DEGREES, OpenCat walking order:
      [LF_sh, RF_sh, RB_sh, LB_sh, LF_k, RF_k, RB_k, LB_k]
  - R r0 r1 r2 r3 r4 r5 r6 r7
      same order as A, but in RADIANS.
  - M d0 d1 d2 d3 d4 d5 d6 d7
      8 target angles in DEGREES, Motion-Imitation policy order:
      [LB_sh, LB_k, LF_sh, LF_k, RB_sh, RB_k, RF_sh, RF_k]
  - N r0 r1 r2 r3 r4 r5 r6 r7
      same as M, but in RADIANS.
  - S
      move to stand pose.
  - P
      power off servo PWM.
  - E
      re-enable PWM and hold current command.
  - L
      print current command angles and calibration.
  - I
      print latest IMU sample:
      I yaw_deg pitch_deg roll_deg yaw_rad pitch_rad roll_rad ax_g ay_g az_g
  - V 0|1
      disable/enable IMU streaming each control tick.
  - D
      print servo diagnostics (pin map, attach state, target and mapped servo angle).
  - T
      toggle between two obvious test poses (for visible movement test).
  - X
      emergency stop (power off PWM + latch fault).
  - C
      clear latched fault.
  - Q
      print fault state.
  - W ms
      set command timeout in milliseconds (stand fallback).
  - O
      print one observation packet for policy input:
      O roll_rad pitch_rad droll_rad_s dpitch_rad_s m0..m7(policy rad)
  - U 0|1
      observation stream off/on.
  - H or ?
      print help.

  Notes:
  - Calibration bytes are loaded from NVS key "config" -> "calib" (same as OpenCat).
  - If motion commands stop, a timeout sends robot back to stand pose.
*/

#include <Arduino.h>
#include <Preferences.h>
#include <Wire.h>
#include "src/PetoiESP32Servo/ESP32Servo.h"
#include "src/icm42670/petoi_icm42670p.h"

namespace {

constexpr int kSerialBaud = 115200;
constexpr uint32_t kControlPeriodMs = 20;      // 50 Hz
constexpr uint32_t kDefaultCommandTimeoutMs = 1200;   // fallback to stand
constexpr float kMaxStepDegPerTick = 8.0f;     // slew-rate limit
constexpr float kRadToDeg = 57.2957795131f;
constexpr float kDegToRad = 0.01745329252f;
constexpr float kNearLimitMarginDeg = 2.0f;    // safety guard margin in OpenCat command space
constexpr uint32_t kNearLimitHoldMs = 400;     // hold near limit for this long -> fault

constexpr int kNumAllJoints = 16;
constexpr int kNumCmdJoints = 8;
constexpr int kPwmNum = 12;

// OpenCat BiBoard V0_2 BITTLE pin mapping.
const uint8_t kPwmPins[kPwmNum] = {
    19, 2,  4,  27,
    33, 5,  15, 14,
    32, 18, 13, 12
};

// Walking joints in OpenCat indexing.
const int kJointIndexByCmd[kNumCmdJoints] = {8, 9, 10, 11, 12, 13, 14, 15};
// Convert walking joint index to PWM channel index: [8..15] -> [4..11].
const int kPwmChannelByCmd[kNumCmdJoints] = {4, 5, 6, 7, 8, 9, 10, 11};

// Motion-Imitation policy order -> OpenCat walking order.
// Policy: [LB_sh, LB_k, LF_sh, LF_k, RB_sh, RB_k, RF_sh, RF_k]
// OpenCat: [LF_sh, RF_sh, RB_sh, LB_sh, LF_k, RF_k, RB_k, LB_k]
void policyToOpenCat(const float in[8], float out[8]) {
  out[0] = in[2];  // LF_sh
  out[1] = in[6];  // RF_sh
  out[2] = in[4];  // RB_sh
  out[3] = in[0];  // LB_sh
  out[4] = in[3];  // LF_k
  out[5] = in[7];  // RF_k
  out[6] = in[5];  // RB_k
  out[7] = in[1];  // LB_k
}

void openCatToPolicy(const float in[8], float out[8]) {
  out[0] = in[3];  // LB_sh
  out[1] = in[7];  // LB_k
  out[2] = in[0];  // LF_sh
  out[3] = in[4];  // LF_k
  out[4] = in[2];  // RB_sh
  out[5] = in[6];  // RB_k
  out[6] = in[1];  // RF_sh
  out[7] = in[5];  // RF_k
}

// OpenCat constants for BITTLE from src/OpenCat.h.
const int8_t kMiddleShift[kNumAllJoints] = {
    0, -90, 0, 0,
    -45, -45, -45, -45,
    55, 55, -55, -55,
    -55, -55, -55, -55
};

const int8_t kRotationDirection[kNumAllJoints] = {
    1, -1, -1, 1,
    1, -1, 1, -1,
    1, -1, -1, 1,
    -1, 1, 1, -1
};

// Angle limits for BITTLE (OpenCat command space in degrees).
const int kAngleLimitMin[kNumAllJoints] = {
    -120, -85, -120, -120,
    -90,  -90, -90,  -90,
    -200, -200, -80, -80,
    -80,  -80, -80,  -80
};

const int kAngleLimitMax[kNumAllJoints] = {
    120, 85, 120, 120,
    60,  60, 90,  90,
    80,  80, 200, 200,
    200, 200, 200, 200
};

// Stand pose in OpenCat walking command order.
const float kStandCmdDeg[kNumCmdJoints] = {
    75.0f, 75.0f, 75.0f, 75.0f, -55.0f, -55.0f, -55.0f, -55.0f
};

// Servo model used for Bittle joints in OpenCat.
ServoModel gServoModel(270, 240, 500, 2500);
Servo gServos[kNumCmdJoints];
Preferences gPrefs;
imu42670p gIcm(Wire, 1);

int8_t gServoCalib[kNumAllJoints] = {0};
float gTargetCmdDeg[kNumCmdJoints] = {0};
float gCurrentCmdDeg[kNumCmdJoints] = {0};

bool gPowerOn = true;
bool gTimeoutStandLatched = false;
bool gSafetyEnabled = true;
bool gFaultLatched = false;
uint32_t gLastControlMs = 0;
uint32_t gLastMotionCmdMs = 0;
uint32_t gLastImuUpdateMs = 0;
uint32_t gCommandTimeoutMs = kDefaultCommandTimeoutMs;
uint32_t gNearLimitStartMs[kNumCmdJoints] = {0};
char gFaultReason[48] = "none";

char gLineBuf[256];
size_t gLineLen = 0;

bool gImuReady = false;
bool gImuStreamEnabled = false;
bool gObsStreamEnabled = false;
float gImuYawDeg = 0.0f;
float gImuPitchDeg = 0.0f;
float gImuRollDeg = 0.0f;
float gImuAxG = 0.0f;
float gImuAyG = 0.0f;
float gImuAzG = 0.0f;
bool gPrintedAttachOk = false;

float clampf(float value, float lo, float hi) {
  if (value < lo) {
    return lo;
  }
  if (value > hi) {
    return hi;
  }
  return value;
}

char *skipSpaces(char *s) {
  while (*s == ' ' || *s == '\t') {
    ++s;
  }
  return s;
}

void loadCalibrationFromNvs() {
  memset(gServoCalib, 0, sizeof(gServoCalib));
  if (!gPrefs.begin("config", true)) {
    Serial.println("WARN: cannot open NVS namespace 'config'. Using zero calib.");
    return;
  }

  const size_t calibLen = gPrefs.getBytesLength("calib");
  if (calibLen >= static_cast<size_t>(kNumAllJoints)) {
    gPrefs.getBytes("calib", gServoCalib, kNumAllJoints);
    Serial.println("Loaded calibration from NVS key 'calib'.");
  } else {
    Serial.println("No valid NVS 'calib' found. Using zero calib.");
  }
  gPrefs.end();
}

void loadIcmOffsetFromNvs() {
  for (int i = 0; i < 3; ++i) {
    gIcm.offset_accel[i] = 0.0f;
    gIcm.offset_gyro[i] = 0.0f;
  }
  if (!gPrefs.begin("config", true)) {
    Serial.println("WARN: cannot open NVS for IMU offsets; using 0.");
    return;
  }

  if (gPrefs.isKey("icm_accel0")) {
    gIcm.offset_accel[0] = gPrefs.getFloat("icm_accel0");
    gIcm.offset_accel[1] = gPrefs.getFloat("icm_accel1");
    gIcm.offset_accel[2] = gPrefs.getFloat("icm_accel2");
    gIcm.offset_gyro[0] = gPrefs.getFloat("icm_gyro0");
    gIcm.offset_gyro[1] = gPrefs.getFloat("icm_gyro1");
    gIcm.offset_gyro[2] = gPrefs.getFloat("icm_gyro2");
    Serial.println("Loaded ICM offsets from NVS.");
  } else {
    Serial.println("No ICM offsets in NVS; using 0.");
  }
  gPrefs.end();
}

bool initImu() {
  Wire.begin();
  delay(5);

  int rc = gIcm.begin();
  if (rc != 0) {
    Serial.print("WARN: ICM42670 begin failed rc=");
    Serial.println(rc);
    return false;
  }
  rc = gIcm.init(200, 2, 250);
  if (rc != 0) {
    Serial.print("WARN: ICM42670 init failed rc=");
    Serial.println(rc);
    return false;
  }

  loadIcmOffsetFromNvs();
  gIcm.getImuGyro();
  gImuReady = true;
  Serial.println("IMU ready");
  return true;
}

void updateImu() {
  if (!gImuReady) {
    return;
  }
  const uint32_t now = millis();
  if (now - gLastImuUpdateMs < kControlPeriodMs) {
    return;
  }
  gLastImuUpdateMs = now;

  gIcm.getImuGyro();
  // Match the OpenCat convention: positive yaw is counterclockwise.
  gImuYawDeg = -gIcm.ypr[0];
  gImuPitchDeg = gIcm.ypr[1];
  gImuRollDeg = gIcm.ypr[2];
  gImuAxG = gIcm.a_real[0];
  gImuAyG = gIcm.a_real[1];
  gImuAzG = gIcm.a_real[2];
}

void printImu() {
  if (!gImuReady) {
    Serial.println("IMU NA");
    return;
  }
  Serial.print("I ");
  Serial.print(gImuYawDeg, 4);
  Serial.print(' ');
  Serial.print(gImuPitchDeg, 4);
  Serial.print(' ');
  Serial.print(gImuRollDeg, 4);
  Serial.print(' ');
  Serial.print(gImuYawDeg * kDegToRad, 6);
  Serial.print(' ');
  Serial.print(gImuPitchDeg * kDegToRad, 6);
  Serial.print(' ');
  Serial.print(gImuRollDeg * kDegToRad, 6);
  Serial.print(' ');
  Serial.print(gImuAxG, 5);
  Serial.print(' ');
  Serial.print(gImuAyG, 5);
  Serial.print(' ');
  Serial.print(gImuAzG, 5);
  Serial.println();
}

void clearNearLimitTimers() {
  for (int i = 0; i < kNumCmdJoints; ++i) {
    gNearLimitStartMs[i] = 0;
  }
}

void triggerFault(const char *reason) {
  for (int i = 0; i < kNumCmdJoints; ++i) {
    gServos[i].writeMicroseconds(0);
  }
  gPowerOn = false;
  gFaultLatched = true;
  strncpy(gFaultReason, reason, sizeof(gFaultReason) - 1);
  gFaultReason[sizeof(gFaultReason) - 1] = '\0';
  clearNearLimitTimers();
  Serial.print("FAULT ");
  Serial.print(gFaultReason);
  Serial.println(" -> PWM OFF");
}

void clearFault() {
  gFaultLatched = false;
  strncpy(gFaultReason, "none", sizeof(gFaultReason) - 1);
  gFaultReason[sizeof(gFaultReason) - 1] = '\0';
  clearNearLimitTimers();
  Serial.println("OK C");
}

void printFaultState() {
  Serial.print("FAULT_STATE latched=");
  Serial.print(gFaultLatched ? "1" : "0");
  Serial.print(" reason=");
  Serial.println(gFaultReason);
}

void printObsPacket() {
  float rollRad = 0.0f;
  float pitchRad = 0.0f;
  float dRollRad = 0.0f;
  float dPitchRad = 0.0f;
  if (gImuReady) {
    rollRad = gImuRollDeg * kDegToRad;
    pitchRad = gImuPitchDeg * kDegToRad;
    dRollRad = gIcm.gx_real * kDegToRad;
    dPitchRad = gIcm.gy_real * kDegToRad;
  }

  float policyDeg[8];
  openCatToPolicy(gCurrentCmdDeg, policyDeg);

  Serial.print("O ");
  Serial.print(rollRad, 6);
  Serial.print(' ');
  Serial.print(pitchRad, 6);
  Serial.print(' ');
  Serial.print(dRollRad, 6);
  Serial.print(' ');
  Serial.print(dPitchRad, 6);
  for (int i = 0; i < 8; ++i) {
    Serial.print(' ');
    Serial.print(policyDeg[i] * kDegToRad, 6);
  }
  Serial.println();
}

void printBoardInfo() {
  Serial.println("BOARD: BiBoard V0_2 (hardcoded)");
  Serial.print("PWM pins[0..11]:");
  for (int i = 0; i < kPwmNum; ++i) {
    Serial.print(' ');
    Serial.print(kPwmPins[i]);
  }
  Serial.println();
}

void ensureServosAttached() {
  static bool timersAllocated = false;
  if (!timersAllocated) {
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);
    timersAllocated = true;
  }

  for (int i = 0; i < kNumCmdJoints; ++i) {
    if (!gServos[i].attached()) {
      const int pwmChannel = kPwmChannelByCmd[i];
      gServos[i].attach(kPwmPins[pwmChannel], &gServoModel);
      delay(2);
      if (!gServos[i].attached()) {
        Serial.print("ERR attach cmd=");
        Serial.print(i);
        Serial.print(" pwm=");
        Serial.print(pwmChannel);
        Serial.print(" pin=");
        Serial.println(kPwmPins[pwmChannel]);
      }
    }
  }
  if (!gPrintedAttachOk) {
    bool allAttached = true;
    for (int i = 0; i < kNumCmdJoints; ++i) {
      allAttached = allAttached && gServos[i].attached();
    }
    if (allAttached) {
      Serial.println("Servo attach OK (8 walking joints)");
      gPrintedAttachOk = true;
    }
  }
  gPowerOn = true;
}

void powerOffServos() {
  for (int i = 0; i < kNumCmdJoints; ++i) {
    gServos[i].writeMicroseconds(0);
  }
  gPowerOn = false;
}

bool checkNearLimitSafety(uint32_t nowMs) {
  if (!gSafetyEnabled || !gPowerOn || gFaultLatched) {
    clearNearLimitTimers();
    return false;
  }
  for (int i = 0; i < kNumCmdJoints; ++i) {
    const int jointIndex = kJointIndexByCmd[i];
    const float minCmd = static_cast<float>(kAngleLimitMin[jointIndex]);
    const float maxCmd = static_cast<float>(kAngleLimitMax[jointIndex]);
    const float cmd = gTargetCmdDeg[i];
    const bool nearMin = (cmd - minCmd) <= kNearLimitMarginDeg;
    const bool nearMax = (maxCmd - cmd) <= kNearLimitMarginDeg;
    if (nearMin || nearMax) {
      if (gNearLimitStartMs[i] == 0) {
        gNearLimitStartMs[i] = nowMs;
      } else if (nowMs - gNearLimitStartMs[i] >= kNearLimitHoldMs) {
        triggerFault("joint_near_limit");
        return true;
      }
    } else {
      gNearLimitStartMs[i] = 0;
    }
  }
  return false;
}

float cmdDegToServoDeg(int cmdIndex, float cmdDeg) {
  const int jointIndex = kJointIndexByCmd[cmdIndex];
  const float limitedCmd = clampf(cmdDeg,
                                  static_cast<float>(kAngleLimitMin[jointIndex]),
                                  static_cast<float>(kAngleLimitMax[jointIndex]));

  const float center = 135.0f;  // P1L angle range center (270 / 2).
  const float zeroPosition =
      center + static_cast<float>(kMiddleShift[jointIndex]) * kRotationDirection[jointIndex];
  const float calibratedZero =
      zeroPosition + static_cast<float>(gServoCalib[jointIndex]) * kRotationDirection[jointIndex];
  const float servoDeg = calibratedZero + limitedCmd * kRotationDirection[jointIndex];
  return clampf(servoDeg, 0.0f, 270.0f);
}

void writeCmdJoint(int cmdIndex, float cmdDeg) {
  const float servoDeg = cmdDegToServoDeg(cmdIndex, cmdDeg);
  gServos[cmdIndex].write(static_cast<int>(roundf(servoDeg)));
}

void setTargetOpenCatDeg(const float deg[8], bool updateMotionTime) {
  for (int i = 0; i < kNumCmdJoints; ++i) {
    const int jointIndex = kJointIndexByCmd[i];
    gTargetCmdDeg[i] = clampf(
        deg[i],
        static_cast<float>(kAngleLimitMin[jointIndex]),
        static_cast<float>(kAngleLimitMax[jointIndex]));
  }
  if (updateMotionTime) {
    gLastMotionCmdMs = millis();
    gTimeoutStandLatched = false;
  }
}

void setTargetPolicyDeg(const float policyDeg[8], bool updateMotionTime) {
  float openCatDeg[8];
  policyToOpenCat(policyDeg, openCatDeg);
  setTargetOpenCatDeg(openCatDeg, updateMotionTime);
}

bool parse8Floats(char *args, float out[8]) {
  for (char *p = args; *p != '\0'; ++p) {
    if (*p == ',') {
      *p = ' ';
    }
  }

  char *cursor = args;
  for (int i = 0; i < 8; ++i) {
    cursor = skipSpaces(cursor);
    if (*cursor == '\0') {
      return false;
    }
    char *endPtr = nullptr;
    out[i] = strtof(cursor, &endPtr);
    if (endPtr == cursor) {
      return false;
    }
    cursor = endPtr;
  }
  return true;
}

void printStatus() {
  Serial.println("STATUS:");
  Serial.print("  power_on=");
  Serial.println(gPowerOn ? "1" : "0");
  Serial.print("  fault_latched=");
  Serial.println(gFaultLatched ? "1" : "0");
  Serial.print("  fault_reason=");
  Serial.println(gFaultReason);
  Serial.print("  safety_enabled=");
  Serial.println(gSafetyEnabled ? "1" : "0");
  Serial.print("  timeout_ms=");
  Serial.println(gCommandTimeoutMs);
  Serial.print("  target_cmd_deg=");
  for (int i = 0; i < kNumCmdJoints; ++i) {
    if (i) {
      Serial.print(' ');
    }
    Serial.print(gTargetCmdDeg[i], 3);
  }
  Serial.println();
  Serial.print("  calib[8..15]=");
  for (int j = 8; j <= 15; ++j) {
    if (j != 8) {
      Serial.print(' ');
    }
    Serial.print(gServoCalib[j]);
  }
  Serial.println();
}

void printServoDiag() {
  Serial.println("SERVO_DIAG:");
  for (int i = 0; i < kNumCmdJoints; ++i) {
    const int jointIndex = kJointIndexByCmd[i];
    const int pwmChannel = kPwmChannelByCmd[i];
    const uint8_t pin = kPwmPins[pwmChannel];
    const float servoDeg = cmdDegToServoDeg(i, gTargetCmdDeg[i]);

    Serial.print("  cmd=");
    Serial.print(i);
    Serial.print(" joint=");
    Serial.print(jointIndex);
    Serial.print(" pwm=");
    Serial.print(pwmChannel);
    Serial.print(" pin=");
    Serial.print(pin);
    Serial.print(" attached=");
    Serial.print(gServos[i].attached() ? "1" : "0");
    Serial.print(" target=");
    Serial.print(gTargetCmdDeg[i], 2);
    Serial.print(" servo=");
    Serial.println(servoDeg, 2);
  }
}

void setTestPoseToggle() {
  static bool flip = false;
  const float poseA[8] = {40.0f, 110.0f, 110.0f, 40.0f, -20.0f, -95.0f, -95.0f, -20.0f};
  const float poseB[8] = {110.0f, 40.0f, 40.0f, 110.0f, -95.0f, -20.0f, -20.0f, -95.0f};
  setTargetOpenCatDeg(flip ? poseA : poseB, true);
  flip = !flip;
}

void printHelp() {
  Serial.println("Bittle Minimal RL Runtime");
  Serial.println("Commands:");
  Serial.println("  A d0..d7  : 8 angles in deg (OpenCat order)");
  Serial.println("  R r0..r7  : 8 angles in rad (OpenCat order)");
  Serial.println("  M d0..d7  : 8 angles in deg (Policy order)");
  Serial.println("  N r0..r7  : 8 angles in rad (Policy order)");
  Serial.println("  S         : stand pose");
  Serial.println("  P         : power off PWM");
  Serial.println("  E         : power on PWM");
  Serial.println("  L         : print status");
  Serial.println("  I         : print IMU sample");
  Serial.println("  V 0|1     : IMU stream off/on");
  Serial.println("  D         : servo diagnostics");
  Serial.println("  T         : toggle test pose");
  Serial.println("  X         : emergency stop (fault latch)");
  Serial.println("  C         : clear fault latch");
  Serial.println("  Q         : print fault state");
  Serial.println("  W ms      : set timeout ms");
  Serial.println("  O         : one-shot policy observation packet");
  Serial.println("  U 0|1     : observation stream off/on");
  Serial.println("  H or ?    : help");
}

void handleLine(char *line) {
  char *p = skipSpaces(line);
  if (*p == '\0') {
    return;
  }

  char cmd = static_cast<char>(toupper(*p));
  ++p;
  p = skipSpaces(p);

  float values[8];
  bool ok = true;

  if (gFaultLatched && cmd != 'C' && cmd != 'Q' && cmd != 'H' && cmd != '?' &&
      cmd != 'I' && cmd != 'V' && cmd != 'L' && cmd != 'O' && cmd != 'U') {
    Serial.println("ERR fault latched, send C");
    return;
  }

  switch (cmd) {
    case 'A':
      ok = parse8Floats(p, values);
      if (ok) {
        ensureServosAttached();
        setTargetOpenCatDeg(values, true);
        Serial.println("OK A");
      }
      break;

    case 'R':
      ok = parse8Floats(p, values);
      if (ok) {
        for (int i = 0; i < 8; ++i) {
          values[i] *= kRadToDeg;
        }
        ensureServosAttached();
        setTargetOpenCatDeg(values, true);
        Serial.println("OK R");
      }
      break;

    case 'M':
      ok = parse8Floats(p, values);
      if (ok) {
        ensureServosAttached();
        setTargetPolicyDeg(values, true);
        Serial.println("OK M");
      }
      break;

    case 'N':
      ok = parse8Floats(p, values);
      if (ok) {
        for (int i = 0; i < 8; ++i) {
          values[i] *= kRadToDeg;
        }
        ensureServosAttached();
        setTargetPolicyDeg(values, true);
        Serial.println("OK N");
      }
      break;

    case 'S':
      ensureServosAttached();
      setTargetOpenCatDeg(kStandCmdDeg, true);
      Serial.println("OK S");
      break;

    case 'P':
      powerOffServos();
      Serial.println("OK P");
      break;

    case 'E':
      ensureServosAttached();
      Serial.println("OK E");
      break;

    case 'L':
      printStatus();
      break;

    case 'D':
      printServoDiag();
      break;

    case 'I':
      updateImu();
      printImu();
      break;

    case 'V':
      if (*p == '1') {
        gImuStreamEnabled = true;
        Serial.println("OK V1");
      } else if (*p == '0') {
        gImuStreamEnabled = false;
        Serial.println("OK V0");
      } else {
        Serial.println("ERR V expects 0 or 1");
      }
      break;

    case 'X':
      triggerFault("manual_estop");
      break;

    case 'C':
      clearFault();
      break;

    case 'Q':
      printFaultState();
      break;

    case 'W': {
      char *endPtr = nullptr;
      long v = strtol(p, &endPtr, 10);
      if (endPtr == p || v < 100 || v > 60000) {
        Serial.println("ERR W expects 100..60000");
      } else {
        gCommandTimeoutMs = static_cast<uint32_t>(v);
        Serial.print("OK W ");
        Serial.println(gCommandTimeoutMs);
      }
      break;
    }

    case 'O':
      updateImu();
      printObsPacket();
      break;

    case 'U':
      if (*p == '1') {
        gObsStreamEnabled = true;
        Serial.println("OK U1");
      } else if (*p == '0') {
        gObsStreamEnabled = false;
        Serial.println("OK U0");
      } else {
        Serial.println("ERR U expects 0 or 1");
      }
      break;

    case 'T':
      ensureServosAttached();
      setTestPoseToggle();
      Serial.println("OK T");
      break;

    case 'H':
    case '?':
      printHelp();
      break;

    default:
      Serial.print("ERR unknown cmd: ");
      Serial.println(cmd);
      ok = false;
      break;
  }

  if (!ok && (cmd == 'A' || cmd == 'R' || cmd == 'M' || cmd == 'N')) {
    Serial.println("ERR parse: expected 8 numeric values");
  }
}

void pollSerial() {
  while (Serial.available() > 0) {
    const char ch = static_cast<char>(Serial.read());
    if (ch == '\r') {
      continue;
    }
    if (ch == '\n') {
      gLineBuf[gLineLen] = '\0';
      handleLine(gLineBuf);
      gLineLen = 0;
      continue;
    }

    if (gLineLen + 1 < sizeof(gLineBuf)) {
      gLineBuf[gLineLen++] = ch;
    } else {
      gLineLen = 0;
      Serial.println("ERR line too long");
    }
  }
}

void updateControl() {
  updateImu();

  const uint32_t now = millis();
  if (now - gLastControlMs < kControlPeriodMs) {
    return;
  }
  gLastControlMs = now;

  if (gPowerOn && (now - gLastMotionCmdMs > gCommandTimeoutMs)) {
    if (!gTimeoutStandLatched) {
      setTargetOpenCatDeg(kStandCmdDeg, false);
      gTimeoutStandLatched = true;
      Serial.println("WARN timeout -> stand");
    }
  }

  if (checkNearLimitSafety(now)) {
    return;
  }

  if (!gPowerOn) {
    return;
  }

  for (int i = 0; i < kNumCmdJoints; ++i) {
    const float delta = gTargetCmdDeg[i] - gCurrentCmdDeg[i];
    const float step = clampf(delta, -kMaxStepDegPerTick, kMaxStepDegPerTick);
    gCurrentCmdDeg[i] += step;
    writeCmdJoint(i, gCurrentCmdDeg[i]);
  }

  if (gImuStreamEnabled) {
    printImu();
  }
  if (gObsStreamEnabled) {
    printObsPacket();
  }
}

}  // namespace

void setup() {
  Serial.begin(kSerialBaud);
  delay(200);
  while (Serial.available() > 0) {
    Serial.read();
  }

  initImu();
  loadCalibrationFromNvs();
  ensureServosAttached();
  printBoardInfo();

  setTargetOpenCatDeg(kStandCmdDeg, true);
  for (int i = 0; i < kNumCmdJoints; ++i) {
    gCurrentCmdDeg[i] = gTargetCmdDeg[i];
    writeCmdJoint(i, gCurrentCmdDeg[i]);
  }

  Serial.println();
  printHelp();
  printStatus();
}

void loop() {
  pollSerial();
  updateControl();
}
