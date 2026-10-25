#!/usr/bin/env python3
"""
waste_sorter.py
Integrated Raspberry Pi waste sorter controller.

Requires:
  - rpi_infer.py (the user's existing TFLite camera-classifier file)
  - waste_classifier_float32_tf.tflite (or set MODEL_PATH below)
  - ADS1115 on I2C (powered from 3.3 V):
        A0 / A1 = capacitive sensors
        A2      = E18-D80NK end-of-belt sensor, through a resistor divider
  - HC-SR04-style ultrasonic sensor in hopper
  - Two digital inductive sensor outputs, through safe 3.3 V interfaces
  - L298N conveyor driver (ENA jumper left ON, controlled by IN1/IN2 only)
  - Hopper servo on GPIO13 and ramp (sorter) servo on GPIO26, both wired directly to the Pi
    and powered from an external 5-6 V supply (common ground with the Pi)

IMPORTANT:
  * Never connect 12 V sensor outputs, the 5 V E18-D80NK output or the ultrasonic ECHO pin
    directly to Pi GPIO or to the ADS1115. The ADS1115 is powered from 3.3 V, so its inputs
    must stay at or below about 3.6 V. Keep the divider on the E18-D80NK output.
  * All grounds must be common where required (Pi, ADS1115, sensors, L298N logic).
  * Test with the motor disconnected and servos unloaded first.
  * Calibrate sensor thresholds, servo angles, and conveyor travel time for your build.
  * KEEP THE END OF THE BELT EMPTY WHEN THE SCRIPT STARTS (the end sensor's idle voltage
    is measured at startup).
  * This is a sequential, one-item-at-a-time example, not a safety-certified controller.
"""

import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

import board
import busio
import cv2
import numpy as np
from gpiozero import AngularServo, DistanceSensor, DigitalInputDevice, DigitalOutputDevice, PWMOutputDevice
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn

from rpi_infer import WasteClassifier, open_camera, check_blank_scene


# --------------------------- CONFIGURATION ---------------------------

MODEL_PATH = Path("waste_classifier_float32_tf.tflite")

# Raspberry Pi BCM GPIO numbering
USE_ENA = False          # False = ENA jumper left ON (motor always enabled, full speed,
                         # controlled only by IN1/IN2). True = jumper removed, PWM speed on ENA.
MOTOR_ENA_PIN = 18       # L298N ENA (only used when USE_ENA = True)
MOTOR_IN1_PIN = 5        # L298N IN1
MOTOR_IN2_PIN = 6        # L298N IN2

INDUCTIVE_1_PIN = 17
INDUCTIVE_2_PIN = 27
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN = 24

# ---- End-of-belt sensor (E18-D80NK) ----
# True  = divider output goes to the ADS1115 (recommended here).
# False = signal goes to GPIO END_SENSOR_PIN through a proper 5 V -> 3.3 V level shifter.
END_SENSOR_USE_ADS = True
END_SENSOR_PIN = 22              # only used when END_SENSOR_USE_ADS = False
END_SENSOR_ADS_CHANNEL = 2       # ADS1115 A2

# ADS mode: the clear (idle) voltage is measured at startup. An item is "present" when the
# voltage drops below idle * this fraction (the NPN output pulls the line to about 0 V).
END_ADS_TRIGGER_FRACTION = 0.5
END_ADS_MIN_IDLE_V = 0.15        # warn at startup if the clear level is below this
END_ADS_CONFIRM_SAMPLES = 2      # consecutive low samples needed (rejects single-sample noise)
END_ADS_DATA_RATE = 860          # ADS1115 samples per second (8..860)

# The end sensor sits 2-3 cm before the ramp. When it sees the item, the ramp turns to the
# bin angle, the belt keeps running this long to carry the item onto the ramp, then stops.
END_RUN_AFTER_DETECT_S = 4.0

# Hopper servo is wired DIRECTLY to a Pi GPIO pin and powered externally.
# GPIO13 is hardware-PWM capable.
HOPPER_SERVO_GPIO = 13

# Ramp (sorter) servo, also wired directly to a Pi GPIO pin with external power.
SORTER_SERVO_GPIO = 26

# Servo pulse range (shared by both servos); tune for your servos
SERVO_MIN_US = 750       # pulse width at 0 deg
SERVO_MAX_US = 2250      # pulse width at 180 deg

# Servo angles; adjust to your actual mechanical positions
HOPPER_CLOSED_DEG = 0
HOPPER_OPEN_DEG = 100
SORT_METAL_DEG = 45
SORT_PLASTIC_DEG = 90
SORT_ORGANIC_DEG = 135
SORT_UNCERTAIN_DEG = 90

# The ultrasonic sensor is assumed to face the item in the hopper.
# Trigger a cycle when the measured distance is at or below this value.
HOPPER_OBJECT_DISTANCE_M = 0.06

# After the item is detected, keep taking capacitive readings for this long
# BEFORE opening the flap. The cycle aborts if the item leaves the hopper.
HOPPER_SETTLE_READ_S = 2.5

# Capacitive sensor thresholds. These are EXAMPLE values only.
# Your sensors read about 2.6 V when idle; calibrate with real samples.
CAP_A0_THRESHOLD_V = 2.50
CAP_A1_THRESHOLD_V = 2.50
CAP_SAMPLE_DELAY_S = 0.05

# If either capacitive reading is below its threshold, the code marks
# "organic-like sensor response". This is only supporting evidence;
# camera classification remains the main non-metal classifier.
CAP_ORGANIC_RULE = "either"  # "either" or "both"

# Set True if the inductive sensor output becomes HIGH on metal detection.
# If the sensor is active-low, set this False.
INDUCTIVE_ACTIVE_HIGH = True

# Conveyor and inference timing
CONVEYOR_SPEED = 0.8             # 0.0 to 1.0 PWM duty cycle (only has an effect if USE_ENA)
CONVEYOR_BRAKE_S = 0.25          # ENA brake time (only used if USE_ENA)
CAMERA_CONFIDENCE_THRESHOLD = 0.70
STABLE_VOTES_REQUIRED = 3
VOTE_WINDOW_SIZE = 5
CLASSIFICATION_TIMEOUT_S = 10.0
CAMERA_FRAME_DELAY_S = 0.03
CONVEYOR_RUN_TIME_S = 30.0  # Safety timeout: the belt ALWAYS stops after this long, even if the
                            # end sensor never fires. Set to None only once the sensor is proven.
RAMP_HOLD_S = 0.8
HOPPER_OPEN_HOLD_S = 0.7
HOPPER_REARM_TIMEOUT_S = 3.0

# For the supplied rpi_infer.py blank-background detector
WHITE_LEVEL = 200
BLACK_LEVEL = 40
BLANK_FRACTION = 0.95

# Camera setup
CAMERA_BACKEND = "auto"  # "auto", "picamera2", or "opencv"
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_HFLIP = False
CAMERA_VFLIP = False

# Live camera output. Shows a window with the prediction overlay.
# With no display (plain SSH), it saves cam_latest.jpg about once a second instead.
SHOW_CAMERA_PREVIEW = True
PREVIEW_WINDOW = "waste sorter camera"
CLASS_LABELS = ["metal", "plastic", "organic"]
HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
_last_preview_save = 0.0
_preview_camera = None   # set in main() so waits can keep the preview alive


# --------------------------- HARDWARE SETUP ---------------------------

i2c = busio.I2C(board.SCL, board.SDA)

ads = ADS1115(i2c)
try:
    ads.data_rate = END_ADS_DATA_RATE   # faster conversions = quicker end-sensor reaction
except Exception as _exc:
    print(f"[ads] Could not set data rate ({_exc}); using the default.")

# The end-sensor thread and the main thread both use the ADS1115, so every read is locked.
# Without this, one thread could change the channel in the middle of the other's conversion.
ads_lock = threading.Lock()

cap_a0 = AnalogIn(ads, 0)
cap_a1 = AnalogIn(ads, 1)


def ads_voltage(channel):
    """Thread-safe single read of an ADS1115 channel, in volts."""
    with ads_lock:
        return channel.voltage


# gpiozero DistanceSensor.distance is in metres, not centimetres.
# The ECHO line MUST be reduced to 3.3 V before it reaches GPIO 24.
ultrasonic = DistanceSensor(
    echo=ULTRASONIC_ECHO_PIN,
    trigger=ULTRASONIC_TRIGGER_PIN,
    max_distance=2.0,
)

# These GPIO inputs must receive safe 3.3 V logic signals.
# Do not connect a 12 V PNP sensor output directly to the Pi.
inductive1 = DigitalInputDevice(INDUCTIVE_1_PIN, pull_up=False)
inductive2 = DigitalInputDevice(INDUCTIVE_2_PIN, pull_up=False)

# ---- End-of-belt sensor state ----
end_latch = threading.Event()       # set when the item has reached the belt end
end_present = threading.Event()     # ADS mode: item currently seen
belt_running = threading.Event()
_stop_end_monitor = threading.Event()

end_sensor = None   # GPIO mode only
end_ch = None       # ADS mode only
END_IDLE_LEVEL = True   # GPIO mode: level seen while clear
END_IDLE_V = None       # ADS mode: voltage seen while clear
END_THRESHOLD_V = None  # ADS mode: below this = item present

if END_SENSOR_USE_ADS:
    end_ch = AnalogIn(ads, END_SENSOR_ADS_CHANNEL)
else:
    # NPN output with weak 5 V pull-up: needs a level shifter, never a direct connection.
    end_sensor = DigitalInputDevice(END_SENSOR_PIN, pull_up=None, active_state=True)

# Latch inductive pulses in hardware callbacks so a short metal pulse is never missed.
# Cleared at the start of each belt run, checked again after the run finishes.
metal_latch = threading.Event()


def _latch_metal():
    metal_latch.set()


for _sensor in (inductive1, inductive2):
    if INDUCTIVE_ACTIVE_HIGH:
        _sensor.when_activated = _latch_metal
    else:
        _sensor.when_deactivated = _latch_metal

# L298N motor controls. With USE_ENA = False the ENA jumper stays on and only IN1/IN2 are used.
motor_enable = PWMOutputDevice(MOTOR_ENA_PIN, frequency=1000, initial_value=0) if USE_ENA else None
motor_in1 = DigitalOutputDevice(MOTOR_IN1_PIN, initial_value=False)
motor_in2 = DigitalOutputDevice(MOTOR_IN2_PIN, initial_value=False)

# Servos direct on GPIO. Signal only: servo power comes from the external supply.
hopper_servo = AngularServo(
    HOPPER_SERVO_GPIO,
    initial_angle=None,
    min_angle=0,
    max_angle=180,
    min_pulse_width=SERVO_MIN_US / 1_000_000,
    max_pulse_width=SERVO_MAX_US / 1_000_000,
)

# Ramp servo direct on GPIO, same arrangement.
sorter_servo = AngularServo(
    SORTER_SERVO_GPIO,
    initial_angle=None,
    min_angle=0,
    max_angle=180,
    min_pulse_width=SERVO_MIN_US / 1_000_000,
    max_pulse_width=SERVO_MAX_US / 1_000_000,
)


# --------------------------- HELPER FUNCTIONS ---------------------------

def set_hopper_angle(angle):
    """Move the hopper servo (direct GPIO) to an angle in degrees (0..180)."""
    hopper_servo.angle = max(0, min(180, float(angle)))


def set_sorter_angle(angle):
    """Move the ramp (sorter) servo (direct GPIO) to an angle in degrees (0..180)."""
    sorter_servo.angle = max(0, min(180, float(angle)))


def show_preview(frame, text, state=""):
    """Show the camera frame with overlay text. Press q in the window to quit."""
    global _last_preview_save
    if not SHOW_CAMERA_PREVIEW or frame is None:
        return
    out = frame.copy()
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if state:
        cv2.putText(out, state, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    if HAS_DISPLAY:
        cv2.imshow(PREVIEW_WINDOW, out)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            raise KeyboardInterrupt
    elif time.monotonic() - _last_preview_save > 1.0:
        cv2.imwrite("cam_latest.jpg", out)
        _last_preview_save = time.monotonic()


def preview_sleep(camera, seconds, text, state=""):
    """Like time.sleep(), but keeps the camera preview updating."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        show_preview(camera.read(), text, state)
        time.sleep(0.02)


def pause(seconds, text="", state=""):
    """time.sleep() replacement that keeps the camera preview updating."""
    if _preview_camera is None or not SHOW_CAMERA_PREVIEW:
        time.sleep(seconds)
    else:
        preview_sleep(_preview_camera, seconds, text, state)


def conveyor_start(speed=CONVEYOR_SPEED):
    """Run conveyor forward. speed (PWM duty 0.0-1.0) only matters when USE_ENA is True."""
    speed = max(0.0, min(1.0, float(speed)))
    motor_in1.on()
    motor_in2.off()
    if motor_enable is not None:
        motor_enable.value = speed
    belt_running.set()


def conveyor_stop():
    """
    Stop the belt with an active brake. L298N: ENA=1 with IN1==IN2 shorts the motor
    (fast stop); ENA=0 just lets it coast.
    """
    belt_running.clear()
    motor_in1.on()
    motor_in2.on()
    if motor_enable is not None:
        motor_enable.value = 1.0      # brake
        time.sleep(CONVEYOR_BRAKE_S)
        motor_enable.value = 0        # then release so the motor isn't held shorted
    # With the ENA jumper on (no ENA wire), IN1 == IN2 already brakes, so just set both low.
    motor_in1.off()
    motor_in2.off()


# ---- End-of-belt sensor ----

def end_level():
    """GPIO mode: raw pin level, True = HIGH."""
    return bool(end_sensor.pin.state)


def _on_end_change():
    """GPIO mode: runs on any edge of the end sensor."""
    level = end_level()
    print(f"[end sensor] edge: pin={'HIGH' if level else 'LOW'}, belt_running={belt_running.is_set()}")
    if level != END_IDLE_LEVEL:
        end_latch.set()


def _end_poll_loop():
    """
    ADS mode: poll the end sensor voltage continuously in its own thread so the detection is
    latched within a few milliseconds, even while the main thread is busy with camera inference.
    """
    hits = 0
    while not _stop_end_monitor.is_set():
        try:
            v = ads_voltage(end_ch)
        except Exception as exc:
            print(f"[end sensor] ADS read error: {exc}")
            time.sleep(0.05)
            continue

        if v < END_THRESHOLD_V:
            hits += 1
        else:
            hits = 0
            end_present.clear()

        if hits >= END_ADS_CONFIRM_SAMPLES:
            if not end_present.is_set():
                end_present.set()
                print(
                    f"[end sensor] item seen: {v:.3f} V (idle {END_IDLE_V:.3f} V, "
                    f"belt_running={belt_running.is_set()})"
                )
            end_latch.set()

        time.sleep(0.001)


def start_end_sensor_monitor():
    """Measure the clear level (belt end must be empty), then start watching the sensor."""
    global END_IDLE_LEVEL, END_IDLE_V, END_THRESHOLD_V

    if END_SENSOR_USE_ADS:
        samples = []
        for _ in range(20):
            samples.append(ads_voltage(end_ch))
            time.sleep(0.02)
        END_IDLE_V = sum(samples) / len(samples)
        END_THRESHOLD_V = END_IDLE_V * END_ADS_TRIGGER_FRACTION
        print(
            f"[end sensor] Idle (clear) = {END_IDLE_V:.3f} V; "
            f"item detected below {END_THRESHOLD_V:.3f} V"
        )
        if END_IDLE_V < END_ADS_MIN_IDLE_V:
            print(
                "[end sensor] WARNING: clear level is very low. Check that nothing is in front "
                "of the sensor, the sensor has 5 V and a common ground, and the divider goes to "
                "ADS1115 A2. Consider a higher-value divider (for example 47k + 47k)."
            )
        threading.Thread(target=_end_poll_loop, daemon=True).start()
    else:
        END_IDLE_LEVEL = end_level()
        print(f"[end sensor] Idle (clear) level = {'HIGH' if END_IDLE_LEVEL else 'LOW'}")
        if not END_IDLE_LEVEL:
            print(
                "[end sensor] WARNING: pin reads LOW with nothing in front. Expected HIGH "
                "when clear. Check the level shifter and the sensor's 5 V supply."
            )
        end_sensor.when_activated = _on_end_change
        end_sensor.when_deactivated = _on_end_change


def item_at_end():
    """True while the end sensor sees an item."""
    if END_SENSOR_USE_ADS:
        return end_present.is_set()
    return end_level() != END_IDLE_LEVEL


def wait_for_item_at_end(camera, started_at, label):
    """
    Keep the belt running until the end sensor sees the item. Returns True if it was seen,
    False if CONVEYOR_RUN_TIME_S passed first. This does NOT stop the belt: the caller turns
    the ramp and runs the belt for END_RUN_AFTER_DETECT_S before stopping.
    """
    while CONVEYOR_RUN_TIME_S is None or time.monotonic() - started_at < CONVEYOR_RUN_TIME_S:
        if end_latch.is_set() or item_at_end():
            return True
        show_preview(camera.read(), f"bin: {label.upper()}", "waiting for item at belt end")
        time.sleep(0.02)
    return False


# ---- Hopper and sensors ----

def hopper_has_item():
    """Check whether the ultrasonic reading is within the configured range."""
    try:
        distance_m = ultrasonic.distance
        return distance_m <= HOPPER_OBJECT_DISTANCE_M, distance_m
    except Exception as exc:
        print(f"[ultrasonic] Read error: {exc}")
        return False, None


def read_capacitive_while_present(duration=HOPPER_SETTLE_READ_S):
    """
    Sample both capacitive channels for `duration` seconds while the item
    stays in the hopper. Returns (avg_a0, avg_a1), or None if the item left.
    """
    v0s, v1s = [], []
    end = time.monotonic() + duration
    gone_since = None

    while time.monotonic() < end:
        present, _ = hopper_has_item()
        if present:
            gone_since = None
        elif gone_since is None:
            gone_since = time.monotonic()
        elif time.monotonic() - gone_since > 0.5:
            return None

        v0s.append(ads_voltage(cap_a0))
        v1s.append(ads_voltage(cap_a1))
        pause(CAP_SAMPLE_DELAY_S, "reading capacitive sensors", "hopper")

    return sum(v0s) / len(v0s), sum(v1s) / len(v1s)


def capacitive_organic_hint(v0, v1):
    """Return True when the configured capacitive threshold rule is met."""
    low0 = v0 < CAP_A0_THRESHOLD_V
    low1 = v1 < CAP_A1_THRESHOLD_V
    if CAP_ORGANIC_RULE == "both":
        return low0 and low1
    return low0 or low1


def metal_detected(sensor):
    """Read an inductive sensor using the configured active polarity."""
    raw = bool(sensor.value)
    return raw if INDUCTIVE_ACTIVE_HIGH else not raw


def classifier_label(index):
    """Keep labels consistent with the supplied rpi_infer.py file."""
    return CLASS_LABELS[index]


def read_camera_prediction(classifier, camera):
    """
    Read one frame. Returns (frame, label, confidence).
    label is None if the scene looks blank or confidence is too low.
    """
    frame = camera.read()
    if frame is None:
        return None, None, 0.0

    blank_kind, _, _ = check_blank_scene(
        frame,
        WHITE_LEVEL,
        BLACK_LEVEL,
        BLANK_FRACTION,
    )
    if blank_kind is not None:
        show_preview(frame, f"blank scene ({blank_kind})", "classifying")
        return frame, None, 0.0

    probabilities, latency_ms = classifier.predict(frame)
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    label = classifier_label(best_index)

    if confidence < CAMERA_CONFIDENCE_THRESHOLD:
        show_preview(
            frame,
            f"low confidence: {label} {confidence:.2f} ({latency_ms:.0f} ms)",
            "classifying",
        )
        return frame, None, confidence

    show_preview(frame, f"{label} {confidence:.2f} ({latency_ms:.0f} ms)", "classifying")
    return frame, label, confidence


def classify_one_item(classifier, camera):
    """
    Collect stable camera predictions while continuously latching either
    inductive sensor. Returns (label, confidence, metal_seen).
    """
    votes = deque(maxlen=VOTE_WINDOW_SIZE)
    confidences = deque(maxlen=VOTE_WINDOW_SIZE)
    metal_seen = False
    start = time.monotonic()

    while time.monotonic() - start < CLASSIFICATION_TIMEOUT_S:
        # The item reached the end of the belt: stop classifying.
        if end_latch.is_set() or item_at_end():
            print("[classification] Item reached belt end before a stable reading.")
            break

        # Latch metal detection so a short sensor pulse is not missed.
        if metal_detected(inductive1) or metal_detected(inductive2):
            metal_seen = True

        frame, label, confidence = read_camera_prediction(classifier, camera)

        if frame is not None and label is not None:
            votes.append(label)
            confidences.append(confidence)

            counts = Counter(votes)
            winning_label, winning_count = counts.most_common(1)[0]
            if (
                winning_count >= STABLE_VOTES_REQUIRED
                and len(votes) >= STABLE_VOTES_REQUIRED
            ):
                matching_confidences = [
                    c for v, c in zip(votes, confidences)
                    if v == winning_label
                ]
                average_confidence = sum(matching_confidences) / len(matching_confidences)
                return winning_label, average_confidence, metal_seen

        time.sleep(CAMERA_FRAME_DELAY_S)

    # Capture any metal signal seen during the timeout too.
    if metal_detected(inductive1) or metal_detected(inductive2):
        metal_seen = True
    return None, 0.0, metal_seen


def choose_sort_class(camera_label, capacitive_hint, metal_seen):
    """
    Metal detection takes priority. Otherwise use camera classification.
    Capacitive readings are logged as supporting evidence only because
    voltage behaviour depends on the actual sensor and waste sample.
    """
    if metal_seen:
        return "metal"

    if camera_label in ("metal", "plastic", "organic"):
        return camera_label

    # Never force an uncertain item into the organic bin solely from
    # an uncalibrated capacitive sensor. Stop safely for manual recovery.
    return None


def ramp_angle_for(label):
    """Ramp angle for a bin label; neutral for None/unknown."""
    return {
        "metal": SORT_METAL_DEG,
        "plastic": SORT_PLASTIC_DEG,
        "organic": SORT_ORGANIC_DEG,
    }.get(label, SORT_UNCERTAIN_DEG)


def route_item(label):
    """Move the ramp to the matching bin position, then return to neutral."""
    angles = {
        "metal": SORT_METAL_DEG,
        "plastic": SORT_PLASTIC_DEG,
        "organic": SORT_ORGANIC_DEG,
    }

    if label not in angles:
        print("[sort] Unknown item: ramp remains at neutral.")
        set_sorter_angle(SORT_UNCERTAIN_DEG)
        return

    set_sorter_angle(angles[label])
    pause(RAMP_HOLD_S, f"ramp -> {label.upper()}", "sorting")
    set_sorter_angle(SORT_UNCERTAIN_DEG)


def release_hopper_item():
    """Open the hopper flap, let one item fall, then close it."""
    print("[hopper] Opening flap.")
    set_hopper_angle(HOPPER_OPEN_DEG)
    pause(HOPPER_OPEN_HOLD_S, "hopper flap open", "releasing item")
    set_hopper_angle(HOPPER_CLOSED_DEG)
    print("[hopper] Closed flap.")


def wait_for_hopper_rearm():
    """
    Wait briefly for the ultrasonic reading to indicate the hopper is clear.
    This avoids repeatedly triggering on the same item. If the sensor sees
    the hopper floor or remaining waste continuously, adjust the threshold
    or replace this with a suitable item-present sensor.
    """
    start = time.monotonic()
    clear_since = None

    while time.monotonic() - start < HOPPER_REARM_TIMEOUT_S:
        present, _distance = hopper_has_item()
        if not present:
            if clear_since is None:
                clear_since = time.monotonic()
            elif time.monotonic() - clear_since >= 0.4:
                return True
        else:
            clear_since = None
        pause(0.1, "waiting for hopper to clear", "re-arming")

    print("[hopper] Did not confirm clear; continuing with one-shot lockout.")
    return False


# --------------------------- MAIN CONTROL LOOP ---------------------------

def main():
    global _preview_camera
    print("Starting waste-sorting controller.")
    print("Check all sensor voltages and motor direction before loading waste.")

    # Open the camera first so the preview is live while the model loads.
    camera = open_camera(
        CAMERA_BACKEND,
        CAMERA_INDEX,
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        CAMERA_HFLIP,
        CAMERA_VFLIP,
    )
    _preview_camera = camera

    loaded = {}

    def _load_model():
        try:
            clf = WasteClassifier(MODEL_PATH, num_threads=4)
            clf.warmup()
            loaded["clf"] = clf
        except Exception as exc:
            loaded["err"] = exc

    loader = threading.Thread(target=_load_model, daemon=True)
    loader.start()
    while loader.is_alive():
        show_preview(camera.read(), "loading model...", "starting up")
        time.sleep(0.02)
    if "err" in loaded:
        camera.release()
        raise loaded["err"]
    classifier = loaded["clf"]

    # Start from safe states.
    conveyor_stop()
    set_hopper_angle(HOPPER_CLOSED_DEG)
    set_sorter_angle(SORT_UNCERTAIN_DEG)

    # Measure the clear level of the end sensor (belt end must be empty) and start watching it.
    start_end_sensor_monitor()

    last_cycle = 0.0
    minimum_cycle_gap_s = 1.0

    try:
        while True:
            present, distance_m = hopper_has_item()

            dist_txt = f"{distance_m:.3f} m" if distance_m is not None else "n/a"
            show_preview(camera.read(), f"hopper distance: {dist_txt}", "waiting for item")

            if not present:
                time.sleep(0.1)
                continue

            if time.monotonic() - last_cycle < minimum_cycle_gap_s:
                time.sleep(0.1)
                continue

            last_cycle = time.monotonic()
            print(f"\n[hopper] Item detected; distance={distance_m:.3f} m")

            # Take capacitive readings for HOPPER_SETTLE_READ_S while the item sits
            # in the hopper. Only after that window does the flap open.
            print(f"[capacitive] Reading for {HOPPER_SETTLE_READ_S:.1f} s before drop...")
            result = read_capacitive_while_present()
            if result is None:
                print("[hopper] Item left before readings finished; aborting cycle.")
                continue
            v0, v1 = result
            cap_hint = capacitive_organic_hint(v0, v1)
            print(
                f"[capacitive] A0={v0:.3f} V, A1={v1:.3f} V, "
                f"organic-like hint={cap_hint}"
            )

            # Release exactly one item.
            release_hopper_item()

            # Start belt and look for this item in the camera.
            print("[conveyor] Starting L298N motor.")
            metal_latch.clear()
            end_latch.clear()
            conveyor_start(CONVEYOR_SPEED)
            conveyor_started_at = time.monotonic()

            camera_label, confidence, metal_seen = classify_one_item(classifier, camera)

            # Also check the latch and the sensors right after classification.
            if (
                metal_latch.is_set()
                or metal_detected(inductive1)
                or metal_detected(inductive2)
            ):
                metal_seen = True

            print(
                f"[classification] camera={camera_label}, "
                f"confidence={confidence:.3f}, metal_sensor={metal_seen}"
            )

            final_label = choose_sort_class(camera_label, cap_hint, metal_seen)
            if final_label is None:
                print("[warning] Classification uncertain. Ramp will stay at neutral. Inspect the item.")
            else:
                print(f"[sort] Selected bin: {final_label.upper()}")

            # The belt keeps running until the end sensor (2-3 cm before the ramp) sees the item.
            seen_at_end = wait_for_item_at_end(
                camera, conveyor_started_at, final_label or "uncertain"
            )

            if not seen_at_end:
                conveyor_stop()
                set_sorter_angle(SORT_UNCERTAIN_DEG)
                print(
                    f"[end sensor] Warning: item not seen at belt end within "
                    f"{CONVEYOR_RUN_TIME_S} s; belt stopped by timeout. Ramp left at neutral."
                )
                wait_for_hopper_rearm()
                continue

            print("[end sensor] Item at belt end.")

            # Final check: the inductive sensors stay armed for the WHOLE belt run,
            # so metal passing them after the camera decision is still caught.
            metal_final = (
                metal_latch.is_set()
                or metal_detected(inductive1)
                or metal_detected(inductive2)
            )
            if metal_final and final_label != "metal":
                print(
                    f"[sensor] Inductive sensor fired during belt run; "
                    f"overriding {final_label} -> METAL."
                )
                final_label = "metal"
            elif final_label == "metal" and not metal_final:
                print(
                    "[sensor] Warning: camera says metal but the inductive "
                    "sensors did not confirm it."
                )

            # Turn the ramp NOW, keep the belt running to carry the item onto it, then stop.
            ramp_text = final_label.upper() if final_label else "UNCERTAIN"
            set_sorter_angle(ramp_angle_for(final_label))
            print(f"[ramp] Turned to {ramp_text}. Belt runs {END_RUN_AFTER_DETECT_S:.1f} s more.")
            pause(END_RUN_AFTER_DETECT_S, f"ramp -> {ramp_text}", "carrying item onto ramp")

            conveyor_stop()
            print("[conveyor] Stopped.")

            # Give the item a moment to slide off, then return the ramp to neutral.
            pause(RAMP_HOLD_S, f"ramp -> {ramp_text}", "item sliding off")
            set_sorter_angle(SORT_UNCERTAIN_DEG)
            print("[cycle] Item routed. Conveyor stopped.")

            # Wait for the hopper to clear before allowing another cycle.
            wait_for_hopper_rearm()

            # Brief pause before the next hopper check.
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[shutdown] Ctrl+C received.")

    finally:
        _stop_end_monitor.set()
        conveyor_stop()
        try:
            set_hopper_angle(HOPPER_CLOSED_DEG)
            set_sorter_angle(SORT_UNCERTAIN_DEG)
        except Exception:
            pass

        _preview_camera = None
        camera.release()
        if HAS_DISPLAY and SHOW_CAMERA_PREVIEW:
            cv2.destroyAllWindows()
        hopper_servo.close()
        sorter_servo.close()
        ultrasonic.close()
        inductive1.close()
        inductive2.close()
        if end_sensor is not None:
            end_sensor.close()
        if motor_enable is not None:
            motor_enable.close()
        motor_in1.close()
        motor_in2.close()
        print("[shutdown] Motor stopped and hardware resources released.")


if __name__ == "__main__":
    main()
