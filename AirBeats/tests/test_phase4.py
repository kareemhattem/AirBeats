# Phase 4 unit tests — run with: python tests/test_phase4.py
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from music.mapping import MusicalMapping, SCALES, SCALE_NAMES

print("=" * 60)
print("Phase 4 Unit Tests — Musical Mapping")
print("=" * 60)

# =============================================================================
# Test 1: Scale definitions
# =============================================================================
print()
print("Available scales:", list(SCALES.keys()))
assert "pentatonic_minor" in SCALES
assert len(SCALES["pentatonic_minor"]) == 5
assert len(SCALES["major"]) == 7
print("  PASS: Scale definitions valid")

# =============================================================================
# Test 2: Note computation — pentatonic minor
# =============================================================================
print()
m = MusicalMapping(scale_name="pentatonic_minor", root_note=60, base_octave=0)
m.print_summary()

# Finger 0 → degree 0 → root → C4 = MIDI 60
note0 = m._midi_note_for_finger(0, roll=0.0)
assert note0 == 60, f"Finger 0 should be C4 (60), got {note0}"
print(f"  Finger 0 (root): MIDI {note0} = {m._note_name(note0)}")

# Finger 1 → degree 1 → minor third → Eb4 = MIDI 63
note1 = m._midi_note_for_finger(1, roll=0.0)
assert note1 == 63, f"Finger 1 should be Eb4 (63), got {note1}"
print(f"  Finger 1 (min 3rd): MIDI {note1} = {m._note_name(note1)}")

# Finger 2 → degree 2 → perfect fourth → F4 = MIDI 65
note2 = m._midi_note_for_finger(2, roll=0.0)
assert note2 == 65, f"Finger 2 should be F4 (65), got {note2}"
print(f"  Finger 2 (4th): MIDI {note2} = {m._note_name(note2)}")

# Finger 3 → degree 3 → perfect fifth → G4 = MIDI 67
note3 = m._midi_note_for_finger(3, roll=0.0)
assert note3 == 67, f"Finger 3 should be G4 (67), got {note3}"
print(f"  Finger 3 (5th): MIDI {note3} = {m._note_name(note3)}")
print("  PASS: Pentatonic minor note mapping")

# =============================================================================
# Test 3: Octave shift from roll
# =============================================================================
print()
# Level hand → octave 0
assert m._roll_to_octave(0.0) == 0
# Tilt right 35° → octave +1
assert m._roll_to_octave(35.0) == 1
# Tilt right 65° → octave +2
assert m._roll_to_octave(65.0) == 2
# Tilt left 35° → octave -1
assert m._roll_to_octave(-35.0) == -1
# Tilt left 65° → octave -2
assert m._roll_to_octave(-65.0) == -2
print("  PASS: Roll-to-octave mapping")

# Verify octave actually shifts the note
note_level = m._midi_note_for_finger(0, roll=0.0)
note_up    = m._midi_note_for_finger(0, roll=35.0)
note_down  = m._midi_note_for_finger(0, roll=-35.0)
assert note_up   == note_level + 12, f"Octave +1 should add 12 semitones"
assert note_down == note_level - 12, f"Octave -1 should subtract 12 semitones"
print(f"  Level={note_level}, +1oct={note_up}, -1oct={note_down}")
print("  PASS: Octave shift changes note")

# =============================================================================
# Test 4: Note triggering with hysteresis
# =============================================================================
print()
m.reset()

# Packet with all fingers extended → no notes
p = {"flex_pct": [0.05, 0.05, 0.05, 0.05], "pressure_pct": 0.5,
     "roll": 0, "pitch": 0, "motion_intensity": 0, "gesture_id": -1,
     "gesture_name": "UNKNOWN"}
result = m.update(p)
assert len(result["active_notes"]) == 0, "No notes should trigger when all extended"
print("  PASS: No notes when fingers extended")

# Bend finger 0 past 0.7 → note ON
p["flex_pct"] = [0.8, 0.05, 0.05, 0.05]
result = m.update(p)
assert len(result["active_notes"]) == 1, f"Expected 1 note ON, got {len(result['active_notes'])}"
assert result["active_notes"][0]["finger"] == 0
assert result["active_notes"][0]["midi_note"] == 60
print(f"  Note ON: finger={result['active_notes'][0]['finger']} "
      f"note={result['active_notes'][0]['midi_note']} "
      f"vel={result['active_notes'][0]['velocity']}")
print("  PASS: Note triggers when finger crosses close threshold")

# Finger 0 stays at 0.5 (hysteresis band) → no new events
p["flex_pct"] = [0.5, 0.05, 0.05, 0.05]
result = m.update(p)
assert len(result["active_notes"]) == 0, "No new note events in hysteresis band"
assert len(result["released_notes"]) == 0, "Note should not release in hysteresis band"
print("  PASS: No events in hysteresis band (0.3-0.7)")

# Finger 0 straightens below 0.3 → note OFF
p["flex_pct"] = [0.1, 0.05, 0.05, 0.05]
result = m.update(p)
assert len(result["released_notes"]) == 1, f"Expected 1 note OFF, got {len(result['released_notes'])}"
assert result["released_notes"][0]["finger"] == 0
print(f"  Note OFF: finger={result['released_notes'][0]['finger']}")
print("  PASS: Note releases when finger crosses open threshold")

# =============================================================================
# Test 5: Velocity from pressure
# =============================================================================
print()
m.reset()

# High pressure → high velocity
p = {"flex_pct": [0.8, 0.05, 0.05, 0.05], "pressure_pct": 1.0,
     "roll": 0, "pitch": 0, "motion_intensity": 0, "gesture_id": -1,
     "gesture_name": "UNKNOWN"}
result = m.update(p)
vel_high = result["active_notes"][0]["velocity"]
assert vel_high == 127, f"Max pressure should give velocity 127, got {vel_high}"
print(f"  High pressure (100%) → velocity={vel_high}")

# Low pressure → lower velocity (but at least MIN_VELOCITY)
m.reset()
p["pressure_pct"] = 0.0
result = m.update(p)
vel_low = result["active_notes"][0]["velocity"]
assert vel_low >= 1, f"Min velocity should be >= 1, got {vel_low}"
assert vel_low < vel_high, f"Low pressure should give lower velocity"
print(f"  Low pressure (0%) → velocity={vel_low}")
print("  PASS: Pressure maps to velocity")

# =============================================================================
# Test 6: Pitch bend from pitch angle
# =============================================================================
print()
# Level → no bend
assert m._pitch_to_bend(0.0) == 0, "Level pitch should give 0 bend"
# Full forward → max positive bend
assert m._pitch_to_bend(45.0) == 8191, f"Pitch +45 should give +8191, got {m._pitch_to_bend(45.0)}"
# Full back → max negative bend
assert m._pitch_to_bend(-45.0) == -8191, f"Pitch -45 should give -8191, got {m._pitch_to_bend(-45.0)}"
# Clamped beyond range
assert m._pitch_to_bend(90.0) == 8191, "Pitch beyond range should clamp"
assert m._pitch_to_bend(-90.0) == -8191, "Pitch beyond range should clamp"
print("  PASS: Pitch angle maps to MIDI pitch bend")

# =============================================================================
# Test 7: Gesture-to-mode mapping
# =============================================================================
print()
m.reset()

# CLOSED_FIST → MUTE
p = {"flex_pct": [0.9, 0.9, 0.9, 0.9], "pressure_pct": 0.0,
     "roll": 0, "pitch": 0, "motion_intensity": 0,
     "gesture_id": 1, "gesture_name": "CLOSED_FIST"}
result = m.update(p)
assert result["mode"] == "MUTE", f"CLOSED_FIST should → MUTE, got {result['mode']}"
print("  PASS: CLOSED_FIST → MUTE")

# OPEN_HAND → PLAY
p = {"flex_pct": [0.05, 0.05, 0.05, 0.05], "pressure_pct": 0.0,
     "roll": 0, "pitch": 0, "motion_intensity": 0,
     "gesture_id": 0, "gesture_name": "OPEN_HAND"}
result = m.update(p)
assert result["mode"] == "PLAY", f"OPEN_HAND should → PLAY, got {result['mode']}"
print("  PASS: OPEN_HAND → PLAY")

# MUTE mode prevents note triggering
m.reset()
m._mode = "MUTE"
p = {"flex_pct": [0.8, 0.8, 0.8, 0.8], "pressure_pct": 0.5,
     "roll": 0, "pitch": 0, "motion_intensity": 0,
     "gesture_id": 1, "gesture_name": "CLOSED_FIST"}
result = m.update(p)
assert len(result["active_notes"]) == 0, "MUTE should prevent note triggers"
print("  PASS: MUTE mode blocks note triggers")

# =============================================================================
# Test 8: Scale cycling
# =============================================================================
print()
m = MusicalMapping()
assert m.scale_name == "pentatonic_minor"
s1 = m.cycle_scale()
assert s1 == "pentatonic_major"
s2 = m.cycle_scale()
assert s2 == "major"
print(f"  cycled: pentatonic_minor → {s1} → {s2}")
print("  PASS: Scale cycling works")

# =============================================================================
# Test 9: Modulation from motion intensity
# =============================================================================
print()
m.reset()
p = {"flex_pct": [0.05, 0.05, 0.05, 0.05], "pressure_pct": 0.0,
     "roll": 0, "pitch": 0, "motion_intensity": 1.0,
     "gesture_id": -1, "gesture_name": "UNKNOWN"}
result = m.update(p)
assert result["modulation"] == 127, f"Full motion → mod 127, got {result['modulation']}"
p["motion_intensity"] = 0.0
result = m.update(p)
assert result["modulation"] == 0, f"No motion → mod 0, got {result['modulation']}"
p["motion_intensity"] = 0.5
result = m.update(p)
assert 60 <= result["modulation"] <= 70, f"Half motion → mod ~63, got {result['modulation']}"
print("  PASS: Motion intensity maps to modulation")

# =============================================================================
# Test 10: Multiple fingers = chord
# =============================================================================
print()
m.reset()
p = {"flex_pct": [0.8, 0.8, 0.8, 0.8], "pressure_pct": 0.7,
     "roll": 0, "pitch": 0, "motion_intensity": 0,
     "gesture_id": -1, "gesture_name": "UNKNOWN"}
result = m.update(p)
assert len(result["active_notes"]) == 4, f"4 fingers bent → 4 notes, got {len(result['active_notes'])}"
notes = [n["midi_note"] for n in result["active_notes"]]
print(f"  Chord: {notes} = {[m._note_name(n) for n in notes]}")
print("  PASS: Multiple fingers trigger chord")

print()
print("=" * 60)
print("All Phase 4 tests passed.")
print("=" * 60)
