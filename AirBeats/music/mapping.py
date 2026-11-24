# =============================================================================
# AirBeats — Musical Mapping
# music/mapping.py
#
# Converts the processed + fused sensor state into musical control events.
#
# This module does NOT generate sound. It produces a MusicalState dict that
# describes what notes should be playing, at what velocity, with what pitch
# bend and modulation. Phase 5 (Sound Generation) consumes this output.
#
# ─────────────────────────────────────────────────────────────────────────────
# Mapping overview
# ─────────────────────────────────────────────────────────────────────────────
#
# Sensor          →  Musical parameter        →  Range / unit
# ────────────────┼──────────────────────────┼─────────────────────────────
# Flex 1 (finger) →  Note trigger + degree    →  Scale degree 0
# Flex 2          →  Note trigger + degree    →  Scale degree 1
# Flex 3          →  Note trigger + degree    →  Scale degree 2
# Flex 4          →  Note trigger + degree    →  Scale degree 3
# Pressure        →  Velocity (loudness)      →  0–127 (MIDI velocity)
# Roll            →  Octave shift             →  -2 to +2 octaves from base
# Pitch           →  Pitch bend               →  -8192 to +8191 (MIDI bend)
# Gesture         →  Performance mode         →  PLAY / MUTE / RELEASE_ALL / TREMOLO
# Motion intensity→  Modulation depth        →  0–127 (MIDI CC 1)
#
# ─────────────────────────────────────────────────────────────────────────────
# Note triggering with hysteresis
# ─────────────────────────────────────────────────────────────────────────────
# Each finger has two thresholds:
#   trigger_on  = GESTURE_CLOSE_THRESHOLD (0.7) — bend past this → note ON
#   trigger_off = GESTURE_OPEN_THRESHOLD  (0.3) — straighten past this → note OFF
# The gap between 0.3 and 0.7 is the hysteresis band. It prevents rapid
# on/off oscillation when the finger is at a borderline position.
# =============================================================================

import config


# =============================================================================
# Scales
# =============================================================================
# Each scale is defined as a list of semitone offsets from the root note.
# The root note is a MIDI note number (e.g. 60 = middle C).
#
# Pentatonic minor:  [0, 3, 5, 7, 10] — no semitones, always sounds good
# Major:            [0, 2, 4, 5, 7, 9, 11] — the "happy" scale
# Natural minor:     [0, 2, 3, 5, 7, 8, 10, 12] — the "sad" scale
#
# We repeat each scale pattern across the full MIDI range (0-127) so any
# scale degree index maps to a valid MIDI note number.

SCALES = {
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":    [0, 2, 3, 5, 7, 8, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
}

SCALE_NAMES = list(SCALES.keys())

# Default scale order for the cycle keyboard command
_SCALE_CYCLE = ["pentatonic_minor", "pentatonic_major", "major", "natural_minor", "blues"]


# =============================================================================
# Musical mapping engine
# =============================================================================

class MusicalMapping:
    """
    Maps sensor state to musical control events.

    Usage:
        mapping = MusicalMapping()
        # Optionally change scale/octave at runtime:
        mapping.set_scale("major")
        mapping.set_base_octave(5)

        # In the acquisition loop, after fusion + gesture recognition:
        musical_state = mapping.update(processed_packet)
        processed_packet["musical_state"] = musical_state
    """

    def __init__(self,
                 scale_name: str = None,
                 root_note: int = None,
                 base_octave: int = None):
        """
        Parameters
        ----------
        scale_name : str
            Name of the scale (see SCALES dict). Default from config.
        root_note : int
            MIDI note number of the root (0-127). 60 = middle C.
            Default from config.
        base_octave : int
            Octave offset applied to all notes (0 = root octave).
            Default from config.
        """
        self._scale_name  = scale_name  or config.DEFAULT_SCALE
        self._root_note   = root_note  if root_note is not None else config.ROOT_NOTE_MIDI
        self._base_octave  = base_octave if base_octave is not None else config.BASE_OCTAVE
        self._scale_pattern = SCALES[self._scale_name]

        # Finger-to-scale-degree mapping
        # Each flex sensor index maps to a position in the scale.
        # Index 0 = degree 0 (root), index 1 = degree 1, etc.
        self._finger_degrees = list(config.FINGER_SCALE_DEGREES)

        # Hysteresis state: track which fingers are currently "on"
        self._finger_active = [False, False, False, False]

        # Velocity captured at note-on time (held until note-off)
        self._finger_velocity = [0, 0, 0, 0]

        # Previous octave for change detection
        self._prev_octave_shift = 0

        # Pitch bend smoothing
        self._pitch_bend = 0   # MIDI pitch bend range: -8192 to +8191

        # Performance mode
        self._mode = "PLAY"

        # Scale cycle index for keyboard cycling
        self._scale_index = _SCALE_CYCLE.index(self._scale_name) \
            if self._scale_name in _SCALE_CYCLE else 0

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    @property
    def scale_name(self) -> str:
        return self._scale_name

    @property
    def root_note(self) -> int:
        return self._root_note

    @property
    def base_octave(self) -> int:
        return self._base_octave

    def set_scale(self, scale_name: str):
        if scale_name in SCALES:
            self._scale_name = scale_name
            self._scale_pattern = SCALES[scale_name]
            print(f"[Mapping] Scale: {scale_name} ({len(self._scale_pattern)} notes)")

    def set_root_note(self, midi_note: int):
        self._root_note = max(0, min(127, int(midi_note)))
        print(f"[Mapping] Root note: {self._root_note} (MIDI)")

    def set_base_octave(self, octave: int):
        self._base_octave = max(0, min(8, int(octave)))
        print(f"[Mapping] Base octave: {self._base_octave}")

    def cycle_scale(self) -> str:
        """Advance to the next scale in the cycle. Returns the new scale name."""
        self._scale_index = (self._scale_index + 1) % len(_SCALE_CYCLE)
        new_name = _SCALE_CYCLE[self._scale_index]
        self.set_scale(new_name)
        return new_name

    def reset(self):
        """Clear all note states (e.g. when switching scales)."""
        self._finger_active   = [False, False, False, False]
        self._finger_velocity  = [0, 0, 0, 0]
        self._prev_octave_shift = 0
        self._pitch_bend       = 0
        self._mode              = "PLAY"

    # -------------------------------------------------------------------------
    # Core mapping logic
    # -------------------------------------------------------------------------

    def update(self, packet: dict) -> dict:
        """
        Map one processed packet to a MusicalState dict.

        Returns:
        {
            "active_notes":  [{"finger": 0, "midi_note": 60, "velocity": 100}, ...],
            "released_notes": [{"finger": 0, "midi_note": 60}, ...],
            "pitch_bend":     int (-8192 to +8191, 0 = no bend),
            "modulation":     int (0-127, MIDI CC 1),
            "octave_shift":   int (current octave offset from base),
            "mode":           str ("PLAY" / "MUTE" / "TREMOLO"),
            "scale":          str (current scale name),
            "root_midi":      int (root note MIDI number),
        }
        """
        flex_pct     = packet.get("flex_pct") or [0, 0, 0, 0]
        pressure_pct  = packet.get("pressure_pct", 0.0)
        roll          = packet.get("roll", 0.0)
        pitch         = packet.get("pitch", 0.0)
        motion        = packet.get("motion_intensity", 0.0)
        gesture_id    = packet.get("gesture_id", -1)
        gesture_name  = packet.get("gesture_name", "UNKNOWN")

        active_notes   = []
        released_notes = []

        # --- 1. Performance mode from gesture ---
        prev_mode = self._mode
        self._mode = self._gesture_to_mode(gesture_id, gesture_name)

        # If mode switched to MUTE, release all active notes
        if self._mode == "MUTE" and prev_mode != "MUTE":
            for i in range(4):
                if self._finger_active[i]:
                    note = self._midi_note_for_finger(i, roll)
                    released_notes.append({"finger": i, "midi_note": note})
                    self._finger_active[i] = False
        elif self._mode == "RELEASE_ALL":
            for i in range(4):
                if self._finger_active[i]:
                    note = self._midi_note_for_finger(i, roll)
                    released_notes.append({"finger": i, "midi_note": note})
                    self._finger_active[i] = False
            self._mode = "PLAY"  # RELEASE_ALL is a one-shot

        # --- 2. Note triggering with hysteresis ---
        if self._mode != "MUTE":
            for i in range(4):
                bend = flex_pct[i] if i < len(flex_pct) else 0.0

                if not self._finger_active[i] and bend >= config.GESTURE_CLOSE_THRESHOLD:
                    # Note ON
                    self._finger_active[i]  = True
                    velocity = self._pressure_to_velocity(pressure_pct)
                    self._finger_velocity[i] = velocity
                    midi_note = self._midi_note_for_finger(i, roll)
                    active_notes.append({
                        "finger": i,
                        "midi_note": midi_note,
                        "velocity": velocity,
                    })

                elif self._finger_active[i] and bend <= config.GESTURE_OPEN_THRESHOLD:
                    # Note OFF
                    self._finger_active[i] = False
                    midi_note = self._midi_note_for_finger(i, roll)
                    released_notes.append({
                        "finger": i,
                        "midi_note": midi_note,
                    })

        # --- 3. Octave shift from roll ---
        octave_shift = self._roll_to_octave(roll)
        if octave_shift != self._prev_octave_shift:
            # Octave changed — need to retrigger active notes at new octave
            # Release current notes first
            for i in range(4):
                if self._finger_active[i]:
                    old_note = self._midi_note_for_finger(i, self._prev_roll_for_octave)
                    released_notes.append({"finger": i, "midi_note": old_note})
            # Retrigger at new octave
            for i in range(4):
                if self._finger_active[i]:
                    new_note = self._midi_note_for_finger(i, roll)
                    active_notes.append({
                        "finger": i,
                        "midi_note": new_note,
                        "velocity": self._finger_velocity[i],
                    })
            self._prev_octave_shift = octave_shift
        self._prev_roll_for_octave = roll

        # --- 4. Pitch bend from pitch angle ---
        self._pitch_bend = self._pitch_to_bend(pitch)

        # --- 5. Modulation from motion intensity ---
        modulation = int(motion * 127)
        modulation = max(0, min(127, modulation))

        return {
            "active_notes":   active_notes,
            "released_notes": released_notes,
            "pitch_bend":     self._pitch_bend,
            "modulation":     modulation,
            "octave_shift":   octave_shift,
            "mode":           self._mode,
            "scale":          self._scale_name,
            "root_midi":      self._root_note,
        }

    # -------------------------------------------------------------------------
    # Internal mapping functions
    # -------------------------------------------------------------------------

    def _midi_note_for_finger(self, finger_index: int, roll: float) -> int:
        """
        Compute the MIDI note number for a finger's scale degree,
        accounting for the current octave shift from roll.

        Formula:
          degree = self._finger_degrees[finger_index]
          semitones_from_root = degree (index into scale pattern)
          octave_offset = base_octave * 12 + octave_shift_from_roll * 12
          midi_note = root_note + octave_offset + scale_pattern[degree % len(pattern)]
                      + 12 * (degree // len(pattern))
        """
        degree = self._finger_degrees[finger_index]
        pattern = self._scale_pattern
        pattern_len = len(pattern)

        # Which octave within the scale this degree falls in
        scale_octave = degree // pattern_len
        index_in_pattern = degree % pattern_len
        semitones = pattern[index_in_pattern] + 12 * scale_octave

        # Octave shift from roll
        octave_shift = self._roll_to_octave(roll)
        total_octave = self._base_octave + octave_shift

        midi = self._root_note + total_octave * 12 + semitones
        return max(0, min(127, midi))

    def _roll_to_octave(self, roll: float) -> int:
        """
        Map roll angle to octave shift.
          roll > +OCTAVE_TILT_THRESHOLD → +1 octave (or +2 for steeper)
          roll < -OCTAVE_TILT_THRESHOLD → -1 octave (or -2 for steeper)
          level → 0

        This lets the user shift octaves by tilting the hand.
        """
        thresh = config.OCTAVE_TILT_THRESHOLD_DEG
        double_thresh = config.OCTAVE_TILT_THRESHOLD_DEG * 2.0

        if roll > double_thresh:
            return 2
        elif roll > thresh:
            return 1
        elif roll < -double_thresh:
            return -2
        elif roll < -thresh:
            return -1
        else:
            return 0

    def _pitch_to_bend(self, pitch: float) -> int:
        """
        Map pitch angle to MIDI pitch bend value.

        MIDI pitch bend is a 14-bit signed value: -8192 to +8191.
        0 = no bend. We map pitch angle linearly:
          pitch = 0  → bend = 0 (no bend)
          pitch = +45 → bend = +8191 (bend up by max semitones)
          pitch = -45 → bend = -8192 (bend down)

        The actual bend range in semitones is set on the synthesizer
        (typically ±2 semitones). This just sends the control value.
        """
        max_bend = 8191
        max_pitch = config.PITCH_BEND_RANGE_DEG  # e.g. 45 degrees
        ratio = max(-1.0, min(1.0, pitch / max_pitch))
        return int(ratio * max_bend)

    def _pressure_to_velocity(self, pressure_pct: float) -> int:
        """
        Map pressure percentage to MIDI velocity (1-127).

        MIDI velocity 0 is reserved for note-off. Minimum playable velocity
        is 1. We map 0% pressure to a minimum velocity and 100% pressure to 127.
        The minimum ensures notes are always audible when triggered.
        """
        min_vel = config.MIN_VELOCITY
        max_vel = 127
        vel = min_vel + pressure_pct * (max_vel - min_vel)
        return max(1, min(127, int(vel)))

    def _gesture_to_mode(self, gesture_id: int, gesture_name: str) -> str:
        """
        Map detected gesture to a performance mode.

        CLOSED_FIST → MUTE (palm mute, like resting hand on guitar strings)
        OPEN_HAND  → PLAY (normal performance)
        SHAKE      → TREMOLO (rapid note repetition effect)
        Other      → keep current mode
        """
        if gesture_id == 1:   # CLOSED_FIST
            return "MUTE"
        elif gesture_id == 11:  # SHAKE
            return "TREMOLO"
        elif gesture_id == 0:  # OPEN_HAND
            return "PLAY"
        else:
            # Keep current mode for other gestures
            return self._mode

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def print_summary(self):
        print(f"[Mapping] Scale: {self._scale_name} ({self._scale_pattern})")
        print(f"[Mapping] Root: MIDI {self._root_note} ({self._note_name(self._root_note)})")
        print(f"[Mapping] Base octave: {self._base_octave}")
        print(f"[Mapping] Finger degrees: {self._finger_degrees}")
        notes = [self._midi_note_for_finger(i, 0) for i in range(4)]
        names = [self._note_name(n) for n in notes]
        print(f"[Mapping] Base octave notes: {notes} = {names}")

    @staticmethod
    def _note_name(midi_note: int) -> str:
        """Convert a MIDI note number to a human-readable name like 'C4'."""
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = midi_note // 12 - 1
        name = names[midi_note % 12]
        return f"{name}{octave}"
