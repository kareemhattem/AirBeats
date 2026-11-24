# =============================================================================
# AirBeats — Sensor Fusion
# sensor_fusion/fusion.py
#
# Estimates hand orientation from the 6-axis IMU using a Complementary Filter,
# and extracts motion features used by the gesture recognizer.
#
# ─────────────────────────────────────────────────────────────────────────────
# Why a Complementary Filter?
# ─────────────────────────────────────────────────────────────────────────────
#
# The accelerometer measures the gravity vector. When the hand is still, you
# can compute tilt angles directly from the direction of gravity:
#
#   roll_accel  = atan2(ay, az)                  [degrees]
#   pitch_accel = atan2(-ax, sqrt(ay^2 + az^2)) [degrees]
#
# These are accurate when stationary, but during motion the accelerometer also
# picks up linear acceleration, making the angles noisy.
#
# The gyroscope measures angular velocity (deg/s). Integrating it over time
# gives angle changes that are accurate in the short term but drift over time
# because any residual bias, after integration, accumulates without bound.
#
# The complementary filter combines both:
#
#   roll  = alpha * (roll_prev  + gx * dt) + (1 - alpha) * roll_accel
#   pitch = alpha * (pitch_prev + gy * dt) + (1 - alpha) * pitch_accel
#
# alpha ~= 0.98: 98% trust the gyro (high-frequency), 2% trust the accel
# (low-frequency drift correction).
#
# This is the standard textbook approach for 6-axis IMU orientation when no
# magnetometer is available. It is simple, computationally cheap, and produces
# roll and pitch. Yaw (rotation around the vertical axis) cannot be estimated
# without a magnetometer because gravity provides no horizontal reference.
#
# ─────────────────────────────────────────────────────────────────────────────
# Confidence weighting
# ─────────────────────────────────────────────────────────────────────────────
#
# When the hand is moving fast, the accelerometer measures both gravity AND
# linear acceleration. The accel-derived angles become unreliable. We detect
# this by checking how far the acceleration magnitude is from 1.0 g (pure
# gravity). If |accel_mag - 1.0| > threshold, we reduce the accelerometer's
# weight in the fusion, effectively trusting the gyro more during fast motion.
# This makes the filter more robust than a fixed-alpha implementation.
# =============================================================================

import math
import numpy as np

import config


class SensorFusion:
    """
    Complementary filter for roll/pitch estimation + motion feature extraction.

    Usage:
        fusion = SensorFusion(sample_rate_hz=50)
        # In the acquisition loop:
        features = fusion.update(processed_packet)
        processed_packet.update(features)
    """

    def __init__(self, sample_rate_hz: float = None, alpha: float = None,
                 accel_confidence_threshold: float = None):
        """
        Parameters
        ----------
        sample_rate_hz : float
            Sampling rate. Must match the Arduino's rate (default 50 Hz).
            Used to compute dt = 1/fs for gyro integration.
        alpha : float
            Complementary filter coefficient. Range (0, 1).
            Higher = trust gyro more. Default 0.98.
        accel_confidence_threshold : float
            If |accel_magnitude - 1.0| exceeds this value, the accelerometer
            is considered unreliable (linear acceleration detected) and its
            weight is reduced toward zero. In g units. Default 0.3 g.
        """
        self._fs      = sample_rate_hz or config.SAMPLE_RATE_HZ
        self._dt      = 1.0 / self._fs
        self._alpha   = alpha if alpha is not None else 0.98
        self._accel_conf_thresh = (
            accel_confidence_threshold if accel_confidence_threshold is not None
            else 0.3
        )

        # Orientation state (degrees)
        self._roll      = 0.0
        self._pitch     = 0.0
        self._yaw_gyro  = 0.0   # Gyro-only integrated yaw (drifts, for reference only)
        self._initialized = False

        # Reference frame (integration bug fix):
        # The IMU's mounting orientation means the resting hand does NOT read
        # roll = 0. With gravity along -Y, the resting roll is ~-86 degrees.
        # Downstream consumers (octave shift, tilt gestures, pitch bend)
        # assume 0 = resting hand. We therefore capture the startup
        # orientation as the reference and subtract it from all outputs,
        # redefining "0 degrees" as the pose the hand was in at startup.
        # The reference is averaged over the first _REF_PACKETS packets
        # (~0.5 s at 50 Hz) to reject a startup motion artifact.
        self._REF_PACKETS = 50
        self._ref_roll_sum  = 0.0
        self._ref_pitch_sum = 0.0
        self._ref_count     = 0
        self._ref_roll      = None   # captured reference, None until set
        self._ref_pitch     = None

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def update(self, packet: dict) -> dict:
        """
        Process one calibrated+filtered packet and return orientation + features.

        Reads (in priority order):
          - accel_filtered  > accel_cal  > accelerometer  (raw)
          - gyro_filtered   > gyro_cal   > gyroscope      (raw)

        Returns a dict with:
          "roll"            : float (degrees, -180 to +180)
          "pitch"           : float (degrees, -90  to +90)
          "yaw_gyro"        : float (degrees, drifts — relative rotation only)
          "gyro_magnitude"  : float (deg/s — total angular speed)
          "accel_magnitude" : float (g — total acceleration, ~1.0 when still)
          "motion_intensity": float (0.0–1.0 — normalised gyro magnitude)
          "is_moving"       : bool  (True if gyro exceeds motion threshold)
        """
        # --- Select best available IMU data ---
        accel = (packet.get("accel_filtered")
                 or packet.get("accel_cal")
                 or packet["accelerometer"])
        gyro  = (packet.get("gyro_filtered")
                 or packet.get("gyro_cal")
                 or packet["gyroscope"])

        ax, ay, az = accel[0], accel[1], accel[2]
        gx, gy, gz = gyro[0],  gyro[1],  gyro[2]

        # --- Step 1: Accelerometer-derived angles ---
        #
        # roll_accel  = atan2(ay, az)
        #   When the hand is flat (palm down), ay ~ -1, az ~ 0  →  roll ~ -90°
        #   When the hand is level,        ay ~ 0,  az ~ -1    →  roll ~ 180°
        #   (sign convention depends on sensor mounting orientation)
        #
        # pitch_accel = atan2(-ax, sqrt(ay^2 + az^2))
        #   When level, ax ~ 0  →  pitch ~ 0°
        #   When tilted forward, ax increases  →  pitch increases
        roll_accel  = math.degrees(math.atan2(ay, az))
        pitch_accel = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))

        # --- Step 2: Accelerometer confidence ---
        #
        # When still, |accel| ~ 1.0 g (just gravity).
        # When moving, |accel| deviates from 1.0 (linear acceleration adds to
        # or subtracts from gravity). The further from 1.0, the less we should
        # trust the accel-derived angles.
        accel_mag = math.sqrt(ax*ax + ay*ay + az*az)
        deviation = abs(accel_mag - 1.0)
        if deviation < self._accel_conf_thresh:
            # Accelerometer is reliable — use the fixed alpha
            accel_weight = 1.0 - self._alpha
        else:
            # Linear acceleration detected — reduce accel trust smoothly
            # The further from 1g, the less we trust it (linear ramp to zero)
            excess = (deviation - self._accel_conf_thresh) / self._accel_conf_thresh
            confidence = max(0.0, 1.0 - excess)
            accel_weight = (1.0 - self._alpha) * confidence

        # --- Step 3: Complementary filter update ---
        if not self._initialized:
            # First call: initialise from accelerometer (no gyro history yet)
            self._roll  = roll_accel
            self._pitch = pitch_accel
            self._initialized = True
        else:
            # Gyro integration (degrees per sample = deg/s * seconds/sample)
            gyro_roll  = self._roll  + gx * self._dt
            gyro_pitch = self._pitch + gy * self._dt

            # Blend: gyro_weight + accel_weight = 1.0
            gyro_weight = 1.0 - accel_weight
            self._roll  = gyro_weight * gyro_roll  + accel_weight * roll_accel
            self._pitch = gyro_weight * gyro_pitch + accel_weight * pitch_accel

        # --- Step 3b: Capture the startup reference frame ---
        # Over the first _REF_PACKETS packets the hand is assumed still
        # (the user starts the app before moving). Average the absolute
        # orientation to lock in the zero reference.
        if self._ref_count < self._REF_PACKETS:
            self._ref_roll_sum  += self._roll
            self._ref_pitch_sum += self._pitch
            self._ref_count     += 1
            if self._ref_count == self._REF_PACKETS:
                self._ref_roll  = self._ref_roll_sum  / self._ref_count
                self._ref_pitch = self._ref_pitch_sum / self._ref_count

        # Output roll/pitch relative to the startup reference.
        # If the reference is not yet captured (first 0.5 s), output 0
        # so downstream consumers see a valid neutral value.
        if self._ref_roll is not None:
            out_roll  = self._roll  - self._ref_roll
            out_pitch = self._pitch - self._ref_pitch
        else:
            out_roll  = 0.0
            out_pitch = 0.0

        # Yaw (gyro-only, drifts — included for relative rotation detection)
        self._yaw_gyro += gz * self._dt

        # --- Step 4: Motion features ---
        gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)

        # Normalised motion intensity (0.0 = still, 1.0 = very fast)
        motion_threshold = config.MOTION_THRESHOLD_DEG_S
        motion_intensity = min(1.0, gyro_mag / (motion_threshold * 2.0))
        is_moving = gyro_mag > motion_threshold

        return {
            "roll"             : out_roll,
            "pitch"            : out_pitch,
            "yaw_gyro"         : self._yaw_gyro,
            "gyro_magnitude"   : gyro_mag,
            "accel_magnitude"  : accel_mag,
            "motion_intensity" : motion_intensity,
            "is_moving"        : is_moving,
        }

    def recapture_reference(self):
        """
        Re-capture the startup reference frame from the current orientation.
        Call this when the hand is held still in the desired neutral pose.

        This resets the reference accumulation and averages the next
        _REF_PACKETS packets of absolute orientation.
        """
        self._ref_roll_sum  = 0.0
        self._ref_pitch_sum = 0.0
        self._ref_count     = 0
        self._ref_roll      = None
        self._ref_pitch     = None
        print(f"[Fusion] Re-capturing reference over next "
              f"{self._REF_PACKETS} packets "
              f"({self._REF_PACKETS / self._fs:.1f} s)...")

    @property
    def reference_captured(self) -> bool:
        return self._ref_roll is not None

    def reset(self):
        """Reset orientation state. Call when the sensor position changes
        abruptly or after recalibration."""
        self._roll       = 0.0
        self._pitch      = 0.0
        self._yaw_gyro   = 0.0
        self._initialized = False
        # Re-capture the reference frame on the next run
        self._ref_roll_sum  = 0.0
        self._ref_pitch_sum = 0.0
        self._ref_count     = 0
        self._ref_roll      = None
        self._ref_pitch     = None

    @property
    def orientation(self) -> dict:
        """Current orientation state (read-only)."""
        return {
            "roll"      : self._roll,
            "pitch"     : self._pitch,
            "yaw_gyro"  : self._yaw_gyro,
            "initialized": self._initialized,
        }
