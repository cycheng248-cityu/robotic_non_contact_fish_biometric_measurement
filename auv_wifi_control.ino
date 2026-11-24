#include <WiFi.h>
#include <WiFiUdp.h>

// ============================================================
// Wi-Fi configuration
// ============================================================

const char* WIFI_SSID = "";
const char* WIFI_PASSWORD = "";

// UDP port on which ESP32 receives motion commands.
const uint16_t UDP_PORT = 4210;

// ============================================================
// TB6612FNG horizontal-thruster wiring
// ============================================================

// Left horizontal thruster: TB6612FNG channel A.
const int LH_IN1 = 18;
const int LH_IN2 = 14;
const int LH_PWM = 13;

// Right horizontal thruster: TB6612FNG channel B.
const int RH_IN1 = 21;
const int RH_IN2 = 22;
const int RH_PWM = 23;

// TB6612FNG standby pin.
// It must be HIGH for either motor channel to operate.
const int STBY_HPIN = 19;

// ============================================================
// ESP32 PWM configuration
// ============================================================

// 20 kHz PWM, using 8-bit duty values from 0 to 255.
const int PWM_FREQUENCY = 20000;
const int PWM_RESOLUTION = 8;

// Bench-test limits. Keep propellers removed or firmly restrain
// the vehicle during initial tests.
const int PWM_MIN_DUTY = 50;
const int PWM_MAX_DUTY = 150;

// ============================================================
// Motion and safety limits
// ============================================================

// Allowed input values in UDP packets.
const float MAX_FORWARD = 0.30;
const float MAX_YAW = 0.30;

// Retained for packet compatibility but intentionally unused.
const float MAX_VERTICAL = 0.00;

// Motor commands below this threshold are treated as STOP.
const float MIN_EFFECTIVE_MOTOR_COMMAND = 0.05;

// Maximum command sent to either individual motor after mixing.
const float MAX_MOTOR_COMMAND = 0.30;

// Stop motors when valid UDP commands stop arriving.
const unsigned long COMMAND_TIMEOUT_MS = 500;

// ============================================================
// Network and command state
// ============================================================

WiFiUDP udp;

char packetBuffer[128];

uint64_t lastSequence = 0;
bool hasAcceptedSequence = false;

unsigned long lastValidCommandTime = 0;
bool watchdogStopped = true;

// ============================================================
// Utilities
// ============================================================

float clampValue(
  float value,
  float minimum,
  float maximum
) {
  if (value < minimum) {
    return minimum;
  }

  if (value > maximum) {
    return maximum;
  }

  return value;
}

// ============================================================
// Motor functions
// ============================================================

void setOneMotor(
  int in1,
  int in2,
  int pwmPin,
  float command
) {
  command = clampValue(
    command,
    -MAX_MOTOR_COMMAND,
    MAX_MOTOR_COMMAND
  );

  float magnitude = fabs(command);

  // Treat small commands as a true stop.
  if (magnitude < MIN_EFFECTIVE_MOTOR_COMMAND) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);

    ledcWrite(
      pwmPin,
      0
    );

    return;
  }

  // Set direction.
  if (command > 0.0f) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  }

  // Map command magnitude:
  // 0.05 -> PWM_MIN_DUTY
  // 0.30 -> PWM_MAX_DUTY
  float fraction = (
    magnitude - MIN_EFFECTIVE_MOTOR_COMMAND
  ) / (
    MAX_MOTOR_COMMAND -
    MIN_EFFECTIVE_MOTOR_COMMAND
  );

  fraction = clampValue(
    fraction,
    0.0f,
    1.0f
  );

  int pwmDuty = int(
    PWM_MIN_DUTY +
    fraction * (
      PWM_MAX_DUTY - PWM_MIN_DUTY
    )
  );

  ledcWrite(
    pwmPin,
    pwmDuty
  );

  Serial.print("PWM GPIO ");
  Serial.print(pwmPin);
  Serial.print(" duty=");
  Serial.println(pwmDuty);
}

void stopHorizontalMotors() {
  // Set directions low.
  digitalWrite(LH_IN1, LOW);
  digitalWrite(LH_IN2, LOW);
  digitalWrite(RH_IN1, LOW);
  digitalWrite(RH_IN2, LOW);

  // Turn PWM output off.
  ledcWrite(LH_PWM, 0);
  ledcWrite(RH_PWM, 0);

  // Disable the TB6612FNG outputs.
  digitalWrite(STBY_HPIN, LOW);
}

void driveHorizontalMotors(
  float forwardCommand,
  float yawCommand
) {
  // Differential mixing:
  // Forward movement: both motors equal.
  // Positive yaw: left motor increases, right motor decreases.
  float leftCommand = forwardCommand + yawCommand;
  float rightCommand = forwardCommand - yawCommand;

  leftCommand = clampValue(
    leftCommand,
    -MAX_MOTOR_COMMAND,
    MAX_MOTOR_COMMAND
  );

  rightCommand = clampValue(
    rightCommand,
    -MAX_MOTOR_COMMAND,
    MAX_MOTOR_COMMAND
  );

  // Enable TB6612FNG before sending PWM.
  digitalWrite(STBY_HPIN, HIGH);

  setOneMotor(
    LH_IN1,
    LH_IN2,
    LH_PWM,
    leftCommand
  );

  setOneMotor(
    RH_IN1,
    RH_IN2,
    RH_PWM,
    rightCommand
  );

  Serial.print("MOTOR COMMANDS | L=");
  Serial.print(leftCommand, 3);
  Serial.print(" | R=");
  Serial.println(rightCommand, 3);
}

// ============================================================
// UDP functions
// ============================================================

bool parseCommandPacket(
  const char* packet,
  uint64_t& sequence,
  float& forwardCommand,
  float& yawCommand,
  float& verticalCommand
) {
  unsigned long long parsedSequence = 0;
  float parsedForward = 0.0f;
  float parsedYaw = 0.0f;
  float parsedVertical = 0.0f;

  int parsedFields = sscanf(
    packet,
    "%llu,%f,%f,%f",
    &parsedSequence,
    &parsedForward,
    &parsedYaw,
    &parsedVertical
  );

  if (parsedFields != 4) {
    return false;
  }

  sequence = (uint64_t)parsedSequence;
  forwardCommand = parsedForward;
  yawCommand = parsedYaw;
  verticalCommand = parsedVertical;

  return true;
}

void sendAcknowledgement(
  IPAddress destinationIp,
  uint16_t destinationPort,
  uint64_t sequence,
  float forwardCommand,
  float yawCommand,
  float verticalCommand
) {
  char acknowledgement[128];

  snprintf(
    acknowledgement,
    sizeof(acknowledgement),
    "ACK:%llu,%.3f,%.3f,%.3f",
    (unsigned long long)sequence,
    forwardCommand,
    yawCommand,
    verticalCommand
  );

  Serial.print("Sending ACK to: ");
  Serial.print(destinationIp);
  Serial.print(":");
  Serial.println(destinationPort);

  udp.beginPacket(
    destinationIp,
    destinationPort
  );

  udp.write(
    (const uint8_t*)acknowledgement,
    strlen(acknowledgement)
  );

  int sendResult = udp.endPacket();

  Serial.print("ACK send result: ");
  Serial.println(sendResult);
}

void processUdpPackets() {
  int packetSize = udp.parsePacket();

  while (packetSize > 0) {
    int bytesRead = udp.read(
      packetBuffer,
      sizeof(packetBuffer) - 1
    );

    if (bytesRead > 0) {
      packetBuffer[bytesRead] = '\0';

      Serial.print("RX: ");
      Serial.println(packetBuffer);

      uint64_t sequence = 0;
      float forwardCommand = 0.0f;
      float yawCommand = 0.0f;
      float verticalCommand = 0.0f;

      bool packetParsed = parseCommandPacket(
        packetBuffer,
        sequence,
        forwardCommand,
        yawCommand,
        verticalCommand
      );

      if (!packetParsed) {
        Serial.println("REJECT: invalid packet format");
      } else if (
        hasAcceptedSequence &&
        sequence <= lastSequence
      ) {
        Serial.print("REJECT: old sequence number: ");
        Serial.println((unsigned long long)sequence);
      } else if (
        fabs(forwardCommand) > MAX_FORWARD ||
        fabs(yawCommand) > MAX_YAW ||
        fabs(verticalCommand) > MAX_VERTICAL
      ) {
        Serial.println("REJECT: command outside allowed limits");
      } else {
        // Accept the packet.
        lastSequence = sequence;
        hasAcceptedSequence = true;
        lastValidCommandTime = millis();
        watchdogStopped = false;

        Serial.print("ACCEPT #");
        Serial.print((unsigned long long)lastSequence);
        Serial.print(" | F=");
        Serial.print(forwardCommand, 3);
        Serial.print(" | Y=");
        Serial.print(yawCommand, 3);
        Serial.print(" | V=");
        Serial.println(verticalCommand, 3);

        // Drive horizontal motors only.
        driveHorizontalMotors(
          forwardCommand,
          yawCommand
        );

        // Reply to the same IP and UDP source port that sent
        // the command packet.
        sendAcknowledgement(
          udp.remoteIP(),
          udp.remotePort(),
          sequence,
          forwardCommand,
          yawCommand,
          verticalCommand
        );
      }
    }

    packetSize = udp.parsePacket();
  }
}

// ============================================================
// Arduino setup
// ============================================================

void setup() {
  Serial.begin(115200);

  pinMode(LH_IN1, OUTPUT);
  pinMode(LH_IN2, OUTPUT);
  pinMode(RH_IN1, OUTPUT);
  pinMode(RH_IN2, OUTPUT);
  pinMode(STBY_HPIN, OUTPUT);

  digitalWrite(LH_IN1, LOW);
  digitalWrite(LH_IN2, LOW);
  digitalWrite(RH_IN1, LOW);
  digitalWrite(RH_IN2, LOW);
  digitalWrite(STBY_HPIN, LOW);

  // Newer ESP32 Arduino Core PWM API.
  ledcAttach(
    LH_PWM,
    PWM_FREQUENCY,
    PWM_RESOLUTION
  );

  ledcAttach(
    RH_PWM,
    PWM_FREQUENCY,
    PWM_RESOLUTION
  );

  ledcWrite(LH_PWM, 0);
  ledcWrite(RH_PWM, 0);

  Serial.println();
  Serial.println("ESP32 horizontal thruster controller starting");

  WiFi.mode(WIFI_STA);
  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("Wi-Fi connected. ESP32 IP: ");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);

  Serial.print("UDP listening on port: ");
  Serial.println(UDP_PORT);

  stopHorizontalMotors();

  Serial.println("System ready.");
  Serial.println("Packet format: sequence,forward,yaw,vertical");
}

// ============================================================
// Arduino loop
// ============================================================

void loop() {
  processUdpPackets();

  unsigned long now = millis();

  bool commandTimedOut = (
    now - lastValidCommandTime
  ) > COMMAND_TIMEOUT_MS;

  if (commandTimedOut && !watchdogStopped) {
    stopHorizontalMotors();

    watchdogStopped = true;

    Serial.println("STOP: command timeout");
  }
}