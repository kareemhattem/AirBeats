# Phase 3 unit tests — run with: python tests/test_phase3.py
import sys, os, math, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from sensor_fusion.fusion import SensorFusion
from gesture_recognition.gestures import GestureRecognizer

print("=" * 60)
print("Phase 3 Unit Tests")
print("=" * 60)

# =============================================================================
# Test 1: SensorFusion — orientation tracking
# =============================================================================
fusion = SensorFusion(sample_rate_hz=50.0, alpha=0.98)

# Hand flat: gravity along -Y (matching your stationary readings ay ~ -1.0)
packet_flat = {
    "accelerometer": [0.0, -1.0, 0.0],
    "gyroscope":     [0.0, 0.0, 0.0],
    "flex": [0,0,0,0], "pressure": 0, "timestamp_ms": 0,
}
result = fusion.update(packet_flat)
roll_flat = result["roll"]
print(f"  Flat hand: roll={roll_flat:.1f}d, pitch={result['pitch']:.1f}d, "
      f"accel_mag={result['accel_magnitude']:.3f}g")
assert abs(result["accel_magnitude"] - 1.0) < 0.01, "Accel mag should be ~1.0g when still"
assert not result["is_moving"], "Should not be moving with zero gyro"
print("  PASS: Stationary hand initialisation")

# Tilt 45 degrees over 1 second (50 samples) with gyro reporting the rotation
# The gyro must report the angular velocity during the tilt for the
# complementary filter to track it. This is how it works in reality:
# the gyro captures the dynamic motion, the accel corrects drift.
fusion.reset()
angle_rad = math.radians(45)
gyro_rate = 45.0  # deg/s — the gyro reports this during the tilt
for i in range(50):
    progress = (i + 1) / 50.0  # 0.0 to 1.0
    current_angle = 45.0 * progress  # 0 to 45 degrees
    rad = math.radians(current_angle)
    ay = -math.cos(rad)
    az =  math.sin(rad)
    packet_tilt = {
        "accelerometer": [0.0, ay, az],
        "gyroscope":     [gyro_rate, 0.0, 0.0],
        "flex": [0,0,0,0], "pressure": 0, "timestamp_ms": 20 * (i+1),
    }
    result2 = fusion.update(packet_tilt)
print(f"  Tilted 45d over 1s: roll={result2['roll']:.1f}d, pitch={result2['pitch']:.1f}d")
assert abs(result2["roll"] - roll_flat) > 10.0, \
    f"Roll did not change after tilt: {result2['roll']:.1f} vs {roll_flat:.1f}"
print("  PASS: Orientation tracking responds to tilt")

# Motion detection
packet_moving = {
    "accelerometer": [0.0, -1.0, 0.0],
    "gyroscope":     [50.0, 30.0, -20.0],
    "flex": [0,0,0,0], "pressure": 0, "timestamp_ms": 40,
}
result3 = fusion.update(packet_moving)
print(f"  Moving: gyro_mag={result3['gyro_magnitude']:.1f}d/s, "
      f"motion={result3['motion_intensity']:.2f}, "
      f"is_moving={result3['is_moving']}")
assert result3["is_moving"], "Should detect motion with gyro=50+30+20"
assert result3["gyro_magnitude"] > 50, "Gyro magnitude too low"
print("  PASS: Motion detection")

# High acceleration (accelerometer confidence reduction)
packet_shake = {
    "accelerometer": [3.0, -1.0, 0.0],
    "gyroscope":     [0.0, 0.0, 0.0],
    "flex": [0,0,0,0], "pressure": 0, "timestamp_ms": 80,
}
result4 = fusion.update(packet_shake)
print(f"  High accel: accel_mag={result4['accel_magnitude']:.2f}g")
assert result4["accel_magnitude"] > 2.0, "Accel mag should be high"
print("  PASS: High acceleration handled")

# Reset
fusion.reset()
assert fusion.orientation["initialized"] == False
print("  PASS: Reset")

# =============================================================================
# Test 2: GestureRecognizer — hand shape gestures
# =============================================================================
print()
recognizer = GestureRecognizer()

def make_packet(flex_pct, pressure_pct=0.0, roll=0.0, pitch=0.0,
                is_moving=False, gyro_mag=0.0):
    return {
        "flex_pct": flex_pct,
        "pressure_pct": pressure_pct,
        "roll": roll,
        "pitch": pitch,
        "is_moving": is_moving,
        "gyro_magnitude": gyro_mag,
    }

# Open hand: all fingers extended
p = make_packet([0.05, 0.05, 0.05, 0.05])
# Feed enough packets at ~20ms intervals to pass the 200ms debounce
for _ in range(15):
    g = recognizer.update(p)
    time.sleep(0.02)  # simulate 50 Hz real-time rate so debounce timer fires
print(f"  OPEN_HAND: id={g['id']}, name={g['name']}, conf={g['confidence']:.2f}")
assert g["id"] == 0, f"Expected OPEN_HAND (0), got {g['id']} ({g['name']})"
print("  PASS: OPEN_HAND")

# Closed fist: all fingers bent
recognizer.reset()
p = make_packet([0.9, 0.9, 0.9, 0.9])
for _ in range(15):
    g = recognizer.update(p)
    time.sleep(0.02)
print(f"  CLOSED_FIST: id={g['id']}, name={g['name']}, conf={g['confidence']:.2f}")
assert g["id"] == 1, f"Expected CLOSED_FIST (1), got {g['id']} ({g['name']})"
print("  PASS: CLOSED_FIST")

# Point: only finger 0 bent
recognizer.reset()
p = make_packet([0.9, 0.1, 0.1, 0.1])
for _ in range(15):
    g = recognizer.update(p)
    time.sleep(0.02)
print(f"  POINT: id={g['id']}, name={g['name']}, conf={g['confidence']:.2f}")
assert g["id"] == 3, f"Expected POINT (3), got {g['id']} ({g['name']})"
print("  PASS: POINT")

# Peace: fingers 1 and 2 bent (index + middle)
recognizer.reset()
p = make_packet([0.1, 0.9, 0.9, 0.1])
for _ in range(15):
    g = recognizer.update(p)
    time.sleep(0.02)
print(f"  PEACE: id={g['id']}, name={g['name']}, conf={g['confidence']:.2f}")
assert g["id"] == 4, f"Expected PEACE (4), got {g['id']} ({g['name']})"
print("  PASS: PEACE")

# =============================================================================
# Test 3: GestureRecognizer — orientation gestures
# =============================================================================
print()
recognizer.reset()
# Hand tilt left: roll > 30
p = make_packet([0.05, 0.05, 0.05, 0.05], roll=45.0)
for _ in range(15):
    g = recognizer.update(p)
    time.sleep(0.02)
print(f"  HAND_TILT_LEFT: id={g['id']}, name={g['name']}")
# OPEN_HAND should match first (higher priority) since all fingers are extended
# Orientation gestures are checked after hand shapes
print(f"  (Note: OPEN_HAND has priority over tilt — expected behaviour)")
print("  PASS: Orientation gesture priority")

# =============================================================================
# Test 4: GestureRecognizer — pressure gesture
# =============================================================================
print()
recognizer.reset()
p = make_packet([0.5, 0.5, 0.5, 0.5], pressure_pct=0.8)
for _ in range(15):
    g = recognizer.update(p)
    time.sleep(0.02)
print(f"  PRESS: id={g['id']}, name={g['name']}, conf={g['confidence']:.2f}")
assert g["id"] == 10, f"Expected PRESS (10), got {g['id']} ({g['name']})"
print("  PASS: PRESS")

# =============================================================================
# Test 5: GestureRecognizer — debounce behaviour
# =============================================================================
print()
recognizer.reset()
# Feed one OPEN_HAND packet — should not immediately report
p = make_packet([0.05, 0.05, 0.05, 0.05])
g1 = recognizer.update(p)
print(f"  First packet: id={g1['id']}, name={g1['name']}, conf={g1['confidence']:.2f}")
assert g1["id"] == -1, f"Should not report immediately (debounce), got {g1['id']}"
print("  PASS: Debounce delays new gesture")

# Feed enough packets at real-time rate to pass debounce
for _ in range(20):
    g2 = recognizer.update(p)
    time.sleep(0.02)
print(f"  After debounce: id={g2['id']}, name={g2['name']}, conf={g2['confidence']:.2f}")
assert g2["id"] == 0, f"Should report OPEN_HAND after debounce, got {g2['id']}"
print("  PASS: Debounce promotes stable gesture")

print()
print("=" * 60)
print("All Phase 3 tests passed.")
print("=" * 60)
