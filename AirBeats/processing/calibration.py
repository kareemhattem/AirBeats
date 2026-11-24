# =============================================================================
# AirBeats — Calibration
# processing/calibration.py
#
# Converts raw sensor readings into physically meaningful, normalised values.
#
# Flex sensors  → bend percentage  0.0 (straight) … 1.0 (fully bent)
# Pressure      → pressure level   0.0 (no press)  … 1.0 (maximum)
# IMU accel     → bias-corrected g values
# IMU gyro      → bias-corrected deg/s values
#
# Key design decisions:
#
#   1. Direction-aware flex calibration.
#      ADC values may increase OR decrease with bending depending on how the
#      sensor is physically installed (voltage divider orientation). The
#      calibration stores both endpoints and the formula works correctly in
#      either direction without needing an explicit "invert" flag.
#
#   2. Per-sensor independent calibration.
#      Each flex sensor gets its own CalibrationProfile. No assumption is made
#      that all sensors respond the same way.
#
#   3. Seed values from the two bench experiments you already ran.
#      These are averaged to give a better starting estimate. The user can
#      recalibrate at runtime while wearing the glove to get better values.
#
#   4. IMU bias calibration from a stationary window.
#      The gyroscope bias is the dominant IMU error source at slow movements.
#      We average N samples taken while the hand is still and store the mean
#      as the bias to subtract on every subsequent reading.
#      The accelerometer bias is handled separately: we store the mean of
#      non-dominant axes (the axes that should be near 0 when stationary) and
#      do NOT remove gravity from the dominant axis — that is Phase 3's job.
#
#   5. Profiles can be saved/loaded as JSON so calibration survives restarts.
# =============================================================================

import json
import os
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class FlexCalibrationProfile:
    """
    Calibration parameters for one flex sensor.

    straight_adc : ADC reading when the finger is fully extended (not bent).
    bent_adc     : ADC reading when the finger is fully bent.

    The sign of (bent_adc - straight_adc) captures direction automatically:
      positive → ADC increases when bent  (e.g. Flex 1 in your data)
      negative → ADC decreases when bent  (e.g. Flex 2, 3, 4 in your data)
    """
    name        : str
    straight_adc: float
    bent_adc    : float
    is_calibrated: bool = False

    @property
    def range_adc(self) -> float:
        """bent_adc - straight_adc. Non-zero when calibrated."""
        return self.bent_adc - self.straight_adc

    def to_percent(self, raw_adc: float) -> float:
        """
        Convert a raw ADC reading to bend percentage [0.0, 1.0].

        0.0 = straight, 1.0 = fully bent.
        Values are clamped to [0.0, 1.0] so out-of-range readings
        (noise, sensor inconsistency) do not cause downstream problems.

        Math:
          normalised = (raw - straight) / (bent - straight)

        When bent > straight (ADC increases with bending):
          raw = straight  → (straight - straight)/(bent - straight) = 0.0
          raw = bent      → (bent - straight)/(bent - straight)     = 1.0

        When bent < straight (ADC decreases with bending):
          raw = straight  → 0/(negative number) = 0.0
          raw = bent      → (negative)/(negative) = 1.0
        Both cases map correctly without needing to invert.
        """
        if abs(self.range_adc) < 1.0:
            # Degenerate calibration — both endpoints are the same.
            # Return 0.0 rather than dividing by near-zero.
            return 0.0
        raw_norm = (raw_adc - self.straight_adc) / self.range_adc
        return max(0.0, min(1.0, raw_norm))


@dataclass
class PressureCalibrationProfile:
    """
    Calibration parameters for the FSR pressure sensor.

    zero_adc : ADC reading with no pressure (typically 0 for FSR).
    max_adc  : ADC reading at the maximum expected pressure.
    """
    zero_adc     : float = 0.0
    max_adc      : float = 1023.0
    is_calibrated: bool  = False

    def to_percent(self, raw_adc: float) -> float:
        """Convert raw ADC to pressure level [0.0, 1.0]."""
        span = self.max_adc - self.zero_adc
        if abs(span) < 1.0:
            return 0.0
        raw_norm = (raw_adc - self.zero_adc) / span
        return max(0.0, min(1.0, raw_norm))


@dataclass
class IMUCalibrationProfile:
    """
    Bias offsets for the IMU.

    accel_bias : [bx, by, bz] in g.   Subtracted from every accelerometer reading.
    gyro_bias  : [bx, by, bz] in deg/s. Subtracted from every gyroscope reading.

    The AlashMPU library already applies hardware calibration in firmware.
    This software-level bias corrects any residual offset that the library
    calibration leaves behind, measured from a recorded stationary window.
    """
    accel_bias   : List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gyro_bias    : List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    is_calibrated: bool = False

    def correct_accel(self, ax: float, ay: float, az: float
                      ) -> tuple[float, float, float]:
        """Subtract accelerometer bias. Returns corrected (ax, ay, az)."""
        return (
            ax - self.accel_bias[0],
            ay - self.accel_bias[1],
            az - self.accel_bias[2],
        )

    def correct_gyro(self, gx: float, gy: float, gz: float
                     ) -> tuple[float, float, float]:
        """Subtract gyroscope bias. Returns corrected (gx, gy, gz)."""
        return (
            gx - self.gyro_bias[0],
            gy - self.gyro_bias[1],
            gz - self.gyro_bias[2],
        )


# =============================================================================
# Default seed values from your two bench experiments
# =============================================================================
# Average of Experiment 1 and Experiment 2 to reduce initial error.
# The user will recalibrate at runtime while wearing the glove.
#
# Experiment 1: straight=[329,298,328,315]  bent=[357,277,304,297]
# Experiment 2: straight=[285,289,305,315]  bent=[402,270,360,407]
#
# Note: Flex2 Exp1 bent=277 is LOWER than straight=298 (ADC decreases).
#       Flex2 Exp2 bent=270 is LOWER than straight=289. Consistent direction.
#       The averaged values preserve the correct direction in both cases.

_DEFAULT_FLEX_PROFILES = [
    FlexCalibrationProfile(
        name="Flex1",
        straight_adc=(329 + 285) / 2,   # 307.0
        bent_adc    =(357 + 402) / 2,   # 379.5
        is_calibrated=True,
    ),
    FlexCalibrationProfile(
        name="Flex2",
        straight_adc=(298 + 289) / 2,   # 293.5
        bent_adc    =(277 + 270) / 2,   # 273.5  (bent < straight → ADC decreases)
        is_calibrated=True,
    ),
    FlexCalibrationProfile(
        name="Flex3",
        straight_adc=(328 + 305) / 2,   # 316.5
        bent_adc    =(304 + 360) / 2,   # 332.0
        is_calibrated=True,
    ),
    FlexCalibrationProfile(
        name="Flex4",
        straight_adc=(315 + 315) / 2,   # 315.0
        bent_adc    =(297 + 407) / 2,   # 352.0
        is_calibrated=True,
    ),
]


# =============================================================================
# Calibration manager
# =============================================================================

class Calibration:
    """
    Manages calibration profiles for all sensors and applies them to packets.

    Usage:
        cal = Calibration()
        # Optionally load a saved profile:
        cal.load("calibration.json")

        # Apply to a raw packet from SerialReader:
        processed = cal.apply(raw_packet)

        # Calibrate IMU from a stationary recording:
        cal.calibrate_imu_from_samples(list_of_raw_packets)

        # Save updated profiles:
        cal.save("calibration.json")
    """

    def __init__(self):
        # Flex: deep-copy defaults so mutations don't affect the class-level defaults
        import copy
        self.flex     = copy.deepcopy(_DEFAULT_FLEX_PROFILES)
        self.pressure = PressureCalibrationProfile()
        self.imu      = IMUCalibrationProfile()

    # -------------------------------------------------------------------------
    # Apply calibration to one packet
    # -------------------------------------------------------------------------

    def apply(self, raw_packet: dict) -> dict:
        """
        Apply all calibration profiles to a raw packet.

        Returns a new dict with the same keys as the raw packet PLUS:
          "flex_pct"      : [float, float, float, float]  — bend 0.0–1.0
          "pressure_pct"  : float                         — pressure 0.0–1.0
          "accel_cal"     : [float, float, float]         — bias-corrected g
          "gyro_cal"      : [float, float, float]         — bias-corrected deg/s

        The original raw values are preserved unchanged.
        """
        # Flex bend percentages
        flex_pct = [
            self.flex[i].to_percent(raw_packet["flex"][i])
            for i in range(4)
        ]

        # Pressure percentage
        pressure_pct = self.pressure.to_percent(raw_packet["pressure"])

        # IMU bias correction
        ax, ay, az = self.imu.correct_accel(*raw_packet["accelerometer"])
        gx, gy, gz = self.imu.correct_gyro(*raw_packet["gyroscope"])

        processed = dict(raw_packet)   # shallow copy preserves raw values
        processed["flex_pct"]     = flex_pct
        processed["pressure_pct"] = pressure_pct
        processed["accel_cal"]    = [ax, ay, az]
        processed["gyro_cal"]     = [gx, gy, gz]

        return processed

    # -------------------------------------------------------------------------
    # Runtime recalibration
    # -------------------------------------------------------------------------

    def set_flex_straight(self, sensor_index: int, raw_adc: float):
        """
        Record the current ADC reading as the straight (0%) endpoint for a
        flex sensor. Called while the finger is fully extended.
        """
        self.flex[sensor_index].straight_adc = raw_adc
        self.flex[sensor_index].is_calibrated = False   # Half done until bent set

    def set_flex_bent(self, sensor_index: int, raw_adc: float):
        """
        Record the current ADC reading as the fully bent (100%) endpoint.
        Called while the finger is fully bent.
        """
        self.flex[sensor_index].bent_adc      = raw_adc
        self.flex[sensor_index].is_calibrated = True

    def set_pressure_max(self, raw_adc: float):
        """Record the maximum expected pressure ADC value."""
        self.pressure.max_adc      = raw_adc
        self.pressure.is_calibrated = True

    def calibrate_imu_from_samples(self, raw_packets: list,
                                   correct_accel_bias: bool = False):
        """
        Compute IMU bias from a list of stationary raw packets and store it.

        Parameters
        ----------
        raw_packets : list of dicts from SerialReader
            Should be collected while the hand is completely still.
            Recommended: 2–5 seconds = 100–250 samples at 50 Hz.
        correct_accel_bias : bool
            If True, also computes and removes accelerometer offset.
            Use with care: when the hand is stationary, one accelerometer axis
            carries gravity (~1 g). Removing the mean of that axis would zero
            out gravity, which is wrong for orientation estimation in Phase 3.
            Leave False unless you have a specific reason to remove it.

        Gyroscope bias is always corrected because a stationary gyro should
        read exactly 0 on all axes. Any non-zero mean is pure bias.
        """
        if len(raw_packets) < 10:
            print("[Calibration] Warning: too few samples for IMU calibration "
                  f"({len(raw_packets)}). Need at least 10.")
            return

        gyros  = np.array([p["gyroscope"]     for p in raw_packets])
        accels = np.array([p["accelerometer"] for p in raw_packets])

        # Gyro bias = mean of stationary readings. Stationary gyro → 0 deg/s.
        gyro_bias = gyros.mean(axis=0).tolist()
        self.imu.gyro_bias = gyro_bias

        if correct_accel_bias:
            # Only safe to subtract bias from axes that should be near 0 g.
            # Identify gravity axis as the axis with the largest absolute mean.
            accel_mean = accels.mean(axis=0)
            gravity_axis = int(np.argmax(np.abs(accel_mean)))
            bias = accel_mean.tolist()
            bias[gravity_axis] = 0.0   # Do NOT remove gravity component
            self.imu.accel_bias = bias

        self.imu.is_calibrated = True

        print("[Calibration] IMU calibrated from stationary window:")
        print(f"  Gyro  bias (deg/s): "
              f"X={self.imu.gyro_bias[0]:+.4f}  "
              f"Y={self.imu.gyro_bias[1]:+.4f}  "
              f"Z={self.imu.gyro_bias[2]:+.4f}")
        if correct_accel_bias:
            print(f"  Accel bias (g):     "
                  f"X={self.imu.accel_bias[0]:+.4f}  "
                  f"Y={self.imu.accel_bias[1]:+.4f}  "
                  f"Z={self.imu.accel_bias[2]:+.4f}")

    # -------------------------------------------------------------------------
    # Noise and calibration diagnostics
    # -------------------------------------------------------------------------

    def compute_imu_noise_stats(self, raw_packets: list) -> dict:
        """
        Compute standard deviation of IMU readings from a stationary window.
        Lower std = less noise. Used to characterise the sensor before filtering.

        Returns a dict with keys: accel_std, gyro_std (each a [x,y,z] list).
        """
        if len(raw_packets) < 10:
            return {}

        gyros  = np.array([p["gyroscope"]     for p in raw_packets])
        accels = np.array([p["accelerometer"] for p in raw_packets])

        stats = {
            "accel_std" : accels.std(axis=0).tolist(),
            "gyro_std"  : gyros.std(axis=0).tolist(),
            "accel_mean": accels.mean(axis=0).tolist(),
            "gyro_mean" : gyros.mean(axis=0).tolist(),
            "n_samples" : len(raw_packets),
        }

        print("[Calibration] IMU noise statistics:")
        print(f"  Accel std  (g)     : "
              f"X={stats['accel_std'][0]:.5f}  "
              f"Y={stats['accel_std'][1]:.5f}  "
              f"Z={stats['accel_std'][2]:.5f}")
        print(f"  Gyro  std  (deg/s) : "
              f"X={stats['gyro_std'][0]:.5f}  "
              f"Y={stats['gyro_std'][1]:.5f}  "
              f"Z={stats['gyro_std'][2]:.5f}")
        print(f"  Accel mean (g)     : "
              f"X={stats['accel_mean'][0]:+.5f}  "
              f"Y={stats['accel_mean'][1]:+.5f}  "
              f"Z={stats['accel_mean'][2]:+.5f}")
        print(f"  Gyro  mean (deg/s) : "
              f"X={stats['gyro_mean'][0]:+.5f}  "
              f"Y={stats['gyro_mean'][1]:+.5f}  "
              f"Z={stats['gyro_mean'][2]:+.5f}")

        return stats

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(self, filepath: str):
        """Serialise all calibration profiles to a JSON file."""
        data = {
            "flex"    : [asdict(f) for f in self.flex],
            "pressure": asdict(self.pressure),
            "imu"     : asdict(self.imu),
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"[Calibration] Saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """
        Load calibration profiles from a JSON file.
        Returns True on success, False if the file does not exist or is corrupt.
        """
        if not os.path.isfile(filepath):
            print(f"[Calibration] File not found: {filepath}. Using defaults.")
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            self.flex = [
                FlexCalibrationProfile(**f) for f in data["flex"]
            ]
            self.pressure = PressureCalibrationProfile(**data["pressure"])
            self.imu      = IMUCalibrationProfile(**data["imu"])

            print(f"[Calibration] Loaded from {filepath}")
            return True
        except Exception as e:
            print(f"[Calibration] Load failed ({e}). Using defaults.")
            return False

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def print_summary(self):
        """Print a human-readable summary of all calibration profiles."""
        print("\n[Calibration] --- Profile Summary ---")
        for i, f in enumerate(self.flex):
            direction = "increases" if f.range_adc > 0 else "decreases"
            print(f"  {f.name:6s}: straight={f.straight_adc:.1f}  "
                  f"bent={f.bent_adc:.1f}  "
                  f"range={f.range_adc:+.1f}  "
                  f"(ADC {direction} when bent)  "
                  f"calibrated={f.is_calibrated}")
        print(f"  Pressure: zero={self.pressure.zero_adc:.1f}  "
              f"max={self.pressure.max_adc:.1f}  "
              f"calibrated={self.pressure.is_calibrated}")
        print(f"  IMU: gyro_bias={[f'{v:+.4f}' for v in self.imu.gyro_bias]}  "
              f"accel_bias={[f'{v:+.4f}' for v in self.imu.accel_bias]}  "
              f"calibrated={self.imu.is_calibrated}")
        print()
