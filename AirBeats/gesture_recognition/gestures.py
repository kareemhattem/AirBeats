# =============================================================================
# AirBeats — Gesture Recognition
# gesture_recognition/gestures.py
#
# Rule-based recognition of predefined hand gestures from the combined sensor
# state (flex percentages, pressure, orientation, motion features).
#
# ─────────────────────────────────────────────────────────────────────────────
# Why rule-based instead of machine learning?
# ─────────────────────────────────────────────────────────────────────────────
# The project brief explicitly states: "The gesture recognition system should
# initially use rule-based recognition. Machine learning may be introduced
# later if justified by collected data."
#
# Rule-based recognition is:
#   - Explainable: every gesture decision can be traced to specific thresholds
#   - No training data needed: works immediately with calibrated values
#   - Sufficient: the gesture vocabulary is small (< 10 gestures)
#   - Easy to tune: change a threshold in config.py and see the effect
#
# A machine learning model would be appropriate if we had hundreds of labeled
# recordings per gesture and needed to recognise subtle variations. For a
# graduation project with a defined vocabulary, rules are the right tool.
#
# ─────────────────────────────────────────────────────────────────────────────
# Gesture vocabulary
# ─────────────────────────────────────────────────────────────────────────────
# Each gesture is defined by a set of conditions on the sensor values:
#
#   ID  | Name            | Conditions
#   ----+-----------------+----------------------------------------------------
#   0   | OPEN_HAND       | All flex < open_thresh  (all fingers extended)
#   1   | CLOSED_FIST     | All flex > close_thresh (all fingers bent)
#   2   | PINCH           | Flex1 > close AND Flex2 > close AND Flex3 < open AND Flex4 < open
#   3   | POINT           | Flex1 > close AND Flex2 < open AND Flex3 < open AND Flex4 < open
#   4   | PEACE           | Flex1 > close AND Flex2 > close AND Flex3 < open AND Flex4 < open
#   5   | THUMB_UP        | Flex4 > close AND all other fingers < open
#   6   | HAND_TILT_LEFT  | roll > tilt_thresh
#   7   | HAND_TILT_RIGHT | roll < -tilt_thresh
#   8   | HAND_TILT_FWD   | pitch > tilt_thresh
#   9   | HAND_TILT_BACK | pitch < -tilt_thresh
#   10  | PRESS           | pressure_pct > press_thresh
#   11  | SHAKE           | is_moving AND gyro_magnitude > shake_thresh (sustained)
#   -1  | UNKNOWN         | No gesture matched (idle or transition)
#
# Flex indices:
#   0 = Flex1 (index?), 1 = Flex2, 2 = Flex3, 3 = Flex4
#   Adjust the index assignment based on which finger each sensor is on.
# =============================================================================

import time

import config


# =============================================================================
# Gesture definition
# =============================================================================

class Gesture:
    """A single gesture with an ID, name, and a list of matching conditions."""

    def __init__(self, gid: int, name: str, conditions: list):
        self.id         = gid
        self.name       = name
        self.conditions  = conditions   # List of (description, predicate)

    def matches(self, state: dict) -> bool:
        """Return True if ALL conditions are satisfied."""
        return all(pred(state) for _, pred in self.conditions)


# =============================================================================
# Gesture Recognizer
# =============================================================================

class GestureRecognizer:
    """
    Recognises hand gestures from the processed + fused sensor state.

    Uses a rule-based approach: each gesture has a set of threshold-based
    conditions. The recognizer evaluates all gestures in priority order and
    returns the first match. Temporal smoothing (debouncing) prevents flicker
    during transitions between gestures.

    Usage:
        recognizer = GestureRecognizer()
        # In the acquisition loop, after fusion:
        gesture = recognizer.update(processed_packet)
        processed_packet["gesture_name"] = gesture["name"]
        processed_packet["gesture_id"]   = gesture["id"]
        processed_packet["gesture_confidence"] = gesture["confidence"]
    """

    def __init__(self):
        self._open_thresh    = config.GESTURE_OPEN_THRESHOLD    # < this = extended
        self._close_thresh   = config.GESTURE_CLOSE_THRESHOLD   # > this = bent
        self._tilt_thresh    = config.GESTURE_TILT_THRESHOLD    # degrees
        self._press_thresh   = config.GESTURE_PRESS_THRESHOLD   # 0-1
        self._shake_thresh   = config.SHAKE_THRESHOLD_DEG_S     # deg/s
        self._debounce_ms    = config.GESTURE_DEBOUNCE_MS        # hold time

        # Build gesture list (order = priority)
        self._gestures = self._build_gestures()

        # State
        self._current_gesture_id = -1     # what we are currently reporting
        self._candidate_gesture_id = -1   # what we think we're transitioning to
        self._candidate_start_time = 0.0  # when the candidate first matched
        self._last_update_time    = 0.0

    def _build_gestures(self) -> list:
        """Build the gesture list in priority order."""
        ot = self._open_thresh
        ct = self._close_thresh

        def flex(i, lo, hi):
            """Flex[i] in range [lo, hi). lo=None means no lower bound."""
            def pred(state):
                vals = state.get("flex_pct") or [0, 0, 0, 0]
                v = vals[i] if i < len(vals) else 0.0
                if lo is not None and v < lo:
                    return False
                if hi is not None and v >= hi:
                    return False
                return True
            return pred

        def roll_gt(thr):
            return lambda s: s.get("roll", 0) > thr

        def roll_lt(thr):
            return lambda s: s.get("roll", 0) < -thr

        def pitch_gt(thr):
            return lambda s: s.get("pitch", 0) > thr

        def pitch_lt(thr):
            return lambda s: s.get("pitch", 0) < -thr

        def press_gt(thr):
            return lambda s: s.get("pressure_pct", 0) > thr

        def is_moving(s):
            return s.get("is_moving", False)

        def gyro_mag_gt(thr):
            return lambda s: s.get("gyro_magnitude", 0) > thr

        gestures = [
            # Hand shapes (checked first — higher priority for music mapping)
            Gesture(1, "CLOSED_FIST", [
                ("all fingers bent", lambda s: all(
                    (s.get("flex_pct") or [0,0,0,0])[i] > ct for i in range(4)
                )),
            ]),
            Gesture(2, "PINCH", [
                ("flex0 bent",    flex(0, ct, None)),
                ("flex1 bent",    flex(1, ct, None)),
                ("flex2 extended", flex(2, None, ot)),
                ("flex3 extended", flex(3, None, ot)),
            ]),
            Gesture(3, "POINT", [
                ("flex0 bent",    flex(0, ct, None)),
                ("flex1 extended", flex(1, None, ot)),
                ("flex2 extended", flex(2, None, ot)),
                ("flex3 extended", flex(3, None, ot)),
            ]),
            Gesture(4, "PEACE", [
                ("flex0 extended", flex(0, None, ot)),
                ("flex1 bent",    flex(1, ct, None)),
                ("flex2 bent",    flex(2, ct, None)),
                ("flex3 extended", flex(3, None, ot)),
            ]),
            Gesture(5, "THUMB_UP", [
                ("flex3 bent",    flex(3, ct, None)),
                ("flex0 extended", flex(0, None, ot)),
                ("flex1 extended", flex(1, None, ot)),
                ("flex2 extended", flex(2, None, ot)),
            ]),
            Gesture(0, "OPEN_HAND", [
                ("all fingers extended", lambda s: all(
                    (s.get("flex_pct") or [0,0,0,0])[i] < ot for i in range(4)
                )),
            ]),

            # Hand orientation (checked after shapes)
            Gesture(6,  "HAND_TILT_LEFT",  [("roll > thresh",  roll_gt(self._tilt_thresh))]),
            Gesture(7,  "HAND_TILT_RIGHT", [("roll < -thresh", roll_lt(self._tilt_thresh))]),
            Gesture(8,  "HAND_TILT_FWD",   [("pitch > thresh", pitch_gt(self._tilt_thresh))]),
            Gesture(9,  "HAND_TILT_BACK",  [("pitch < -thresh", pitch_lt(self._tilt_thresh))]),

            # Dynamic gestures
            Gesture(11, "SHAKE", [
                ("is_moving",      is_moving),
                ("gyro_mag high",  gyro_mag_gt(self._shake_thresh)),
            ]),

            # Pressure
            Gesture(10, "PRESS", [
                ("pressure > thresh", press_gt(self._press_thresh)),
            ]),
        ]

        return gestures

    # -------------------------------------------------------------------------
    # Recognition
    # -------------------------------------------------------------------------

    def update(self, packet: dict) -> dict:
        """
        Evaluate all gestures against the current sensor state.
        Returns a dict: {"id": int, "name": str, "confidence": float}

        Debouncing: a new gesture must be sustained for DEBOUNCE_MS before it
        is reported. This prevents rapid flicker during transitions.
        """
        now = time.perf_counter() * 1000.0  # ms

        # Find the first matching gesture
        matched_id   = -1
        matched_name = "UNKNOWN"
        for g in self._gestures:
            if g.matches(packet):
                matched_id   = g.id
                matched_name = g.name
                break

        # --- Debouncing ---
        if matched_id != self._candidate_gesture_id:
            # New candidate gesture appeared
            self._candidate_gesture_id  = matched_id
            self._candidate_start_time = now
            confidence = 0.0
        else:
            # Same candidate as last time — check if held long enough
            held_ms = now - self._candidate_start_time
            if held_ms >= self._debounce_ms and matched_id != self._current_gesture_id:
                # Candidate has been stable long enough — promote to current
                self._current_gesture_id = matched_id
                confidence = 1.0
            elif matched_id == self._current_gesture_id:
                # Still matching the current gesture
                confidence = 1.0
            else:
                # In transition, not yet stable
                confidence = held_ms / self._debounce_ms
                confidence = min(confidence, 0.99)

        return {
            "id"        : self._current_gesture_id,
            "name"      : self._gesture_name(self._current_gesture_id),
            "confidence": confidence,
            "candidate" : matched_name,
        }

    @staticmethod
    def _gesture_name(gid: int) -> str:
        names = {
            0: "OPEN_HAND", 1: "CLOSED_FIST", 2: "PINCH", 3: "POINT",
            4: "PEACE", 5: "THUMB_UP", 6: "HAND_TILT_LEFT",
            7: "HAND_TILT_RIGHT", 8: "HAND_TILT_FWD", 9: "HAND_TILT_BACK",
            10: "PRESS", 11: "SHAKE", -1: "UNKNOWN",
        }
        return names.get(gid, "UNKNOWN")

    def reset(self):
        """Clear all gesture state."""
        self._current_gesture_id    = -1
        self._candidate_gesture_id  = -1
        self._candidate_start_time  = 0.0
