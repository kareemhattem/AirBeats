# Phase 5 unit tests - run with: python tests/test_phase5.py
import sys, os, time, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from music.synthesizer import Synthesizer, Voice, INSTRUMENT_PRESETS, INSTRUMENT_NAMES
from music.midi_engine import MidiEngine
import config

print("=" * 60)
print("Phase 5 Unit Tests - Sound Generation")
print("=" * 60)

# =============================================================================
# Test 1: Voice produces audio
# =============================================================================
print()
inst = INSTRUMENT_PRESETS["piano"]
voice = Voice(midi_note=60, velocity=100, instrument=inst, fs=44100)

samples = voice.render(256)
assert len(samples) == 256, f"Expected 256 samples, got {len(samples)}"
assert np.any(samples != 0), "Voice should produce non-zero audio"
assert samples.dtype == np.float32, f"Expected float32, got {samples.dtype}"
assert np.max(np.abs(samples)) <= 1.0, "Samples should be in [-1, 1]"
print(f"  Voice: 256 samples, max amp={np.max(np.abs(samples)):.4f}")
print("  PASS: Voice produces audio")

# =============================================================================
# Test 2: Note-on then note-off produces envelope shape
# =============================================================================
print()
voice2 = Voice(midi_note=69, velocity=80, instrument=inst, fs=44100)
# Render some attack + decay + sustain
s1 = voice2.render(4410)  # 100ms
amp_during = np.max(np.abs(s1))
assert amp_during > 0.01, f"Note should be audible during sustain, amp={amp_during}"

# Release the note
voice2.note_off()
s2 = voice2.render(44100)  # 1 second of release
amp_after = np.max(np.abs(s2))
assert amp_after < amp_during, "Note should be quieter after release"

# After enough time, voice should be done
assert voice2.is_done, "Voice should be done after release phase"
print(f"  During sustain: amp={amp_during:.4f}, After release: amp={amp_after:.4f}")
print("  PASS: ADSR envelope works")

# =============================================================================
# Test 3: Synthesizer note_on / note_off / all_notes_off
# =============================================================================
print()
synth = Synthesizer()
# Test without starting the audio stream (we can still call note methods)

# Note on
synth.note_on(60, velocity=100)
assert synth.active_note_count == 1, f"Expected 1 active note, got {synth.active_note_count}"
assert 60 in synth._active_notes_set
print(f"  note_on(60): active={synth.active_note_count}")

# Note off
synth.note_off(60)
# Note off sets the voice to release, but the note is removed from the active set
assert synth.active_note_count == 0, f"Expected 0 after note_off, got {synth.active_note_count}"
print(f"  note_off(60): active={synth.active_note_count}")

# Multiple notes (chord)
synth.note_on(60, 100)
synth.note_on(64, 100)
synth.note_on(67, 100)
assert synth.active_note_count == 3, f"Expected 3 active, got {synth.active_note_count}"
print(f"  Chord (C-E-G): active={synth.active_note_count}")

# All notes off
synth.all_notes_off()
assert synth.active_note_count == 0, f"Expected 0 after all_notes_off, got {synth.active_note_count}"
print(f"  all_notes_off: active={synth.active_note_count}")
print("  PASS: Note on/off/all_notes_off")

# =============================================================================
# Test 4: Pitch bend and modulation update voices
# =============================================================================
print()
synth.note_on(60, 100)
synth.set_pitch_bend(4096)  # half up
assert synth._pitch_bend == 4096
synth.set_modulation(64)
assert synth._modulation == 64
# Check that the voice got updated
with synth._lock:
    v = synth._voices[0]
    assert v._pitch_ratio > 1.0, f"Pitch bend up should increase ratio: {v._pitch_ratio}"
    assert v._mod_depth > 0, f"Modulation should be > 0: {v._mod_depth}"
print(f"  Pitch bend: ratio={v._pitch_ratio:.4f} (should be > 1.0)")
print(f"  Modulation: depth={v._mod_depth:.4f} (should be > 0)")
synth.all_notes_off()
print("  PASS: Pitch bend and modulation applied to voices")

# =============================================================================
# Test 5: Instrument cycling
# =============================================================================
print()
synth2 = Synthesizer()
assert synth2.instrument_name == config.DEFAULT_INSTRUMENT
inst1 = synth2.cycle_instrument()
inst2 = synth2.cycle_instrument()
assert inst1 != inst2, "Cycling should change instruments"
print(f"  Cycled: {config.DEFAULT_INSTRUMENT} -> {inst1} -> {inst2}")
print("  PASS: Instrument cycling works")

# =============================================================================
# Test 6: MidiEngine translates musical state
# =============================================================================
print()
# Create a mock synthesizer to test the engine without audio output
class MockSynth:
    def __init__(self):
        self.notes_on = []
        self.notes_off = []
        self.all_off_count = 0
        self.pitch_bend = 0
        self.modulation = 0
        self.is_running = True
        self._instrument_name = "piano"

    def note_on(self, note, vel):
        self.notes_on.append((note, vel))

    def note_off(self, note):
        self.notes_off.append(note)

    def all_notes_off(self):
        self.all_off_count += 1
        self.notes_on.clear()
        self.notes_off.clear()

    def set_pitch_bend(self, b):
        self.pitch_bend = b

    def set_modulation(self, m):
        self.modulation = m

    def stop(self):
        pass

    @property
    def instrument_name(self):
        return self._instrument_name

    @property
    def active_note_count(self):
        return 0

mock = MockSynth()
engine = MidiEngine(mock)

# Test note-on
state = {
    "active_notes": [{"finger": 0, "midi_note": 60, "velocity": 100}],
    "released_notes": [],
    "pitch_bend": 2000,
    "modulation": 50,
    "mode": "PLAY",
}
engine.update(state)
assert mock.notes_on == [(60, 100)], f"Expected note_on(60,100), got {mock.notes_on}"
assert mock.pitch_bend == 2000
assert mock.modulation == 50
print(f"  Note-on: {mock.notes_on}, bend={mock.pitch_bend}, mod={mock.modulation}")
print("  PASS: MidiEngine note-on")

# Test note-off
state = {
    "active_notes": [],
    "released_notes": [{"finger": 0, "midi_note": 60}],
    "pitch_bend": 0,
    "modulation": 0,
    "mode": "PLAY",
}
engine.update(state)
assert 60 in mock.notes_off, f"Expected note_off(60), got {mock.notes_off}"
print(f"  Note-off: {mock.notes_off}")
print("  PASS: MidiEngine note-off")

# Test MUTE mode
mock2 = MockSynth()
engine2 = MidiEngine(mock2)
# First trigger a note
state_play = {
    "active_notes": [{"finger": 0, "midi_note": 60, "velocity": 80}],
    "released_notes": [],
    "pitch_bend": 0, "modulation": 0, "mode": "PLAY",
}
engine2.update(state_play)
assert len(mock2.notes_on) == 1

# Now mute
state_mute = {
    "active_notes": [],
    "released_notes": [],
    "pitch_bend": 0, "modulation": 0, "mode": "MUTE",
}
engine2.update(state_mute)
assert mock2.all_off_count > 0, "MUTE should call all_notes_off"
print(f"  MUTE: all_notes_off called {mock2.all_off_count} time(s)")
print("  PASS: MidiEngine MUTE mode")

# Test MUTE blocks new notes
state_mute2 = {
    "active_notes": [{"finger": 0, "midi_note": 72, "velocity": 100}],
    "released_notes": [],
    "pitch_bend": 0, "modulation": 0, "mode": "MUTE",
}
engine2.update(state_mute2)
# Should NOT have added note 72
assert len(mock2.notes_on) == 0, "MUTE should prevent new notes"
print("  PASS: MUTE blocks new note-on events")

# =============================================================================
# Test 7: No stuck notes
# =============================================================================
print()
mock3 = MockSynth()
engine3 = MidiEngine(mock3)

# Trigger a note
engine3.update({
    "active_notes": [{"finger": 0, "midi_note": 60, "velocity": 100}],
    "released_notes": [], "pitch_bend": 0, "modulation": 0, "mode": "PLAY",
})
assert engine3.active_note_count == 1

# Release it
engine3.update({
    "active_notes": [],
    "released_notes": [{"finger": 0, "midi_note": 60}],
    "pitch_bend": 0, "modulation": 0, "mode": "PLAY",
})
assert engine3.active_note_count == 0, "Note should be released"
print("  PASS: No stuck notes after release")

# Panic
engine3.update({
    "active_notes": [{"finger": 0, "midi_note": 60, "velocity": 100}],
    "released_notes": [], "pitch_bend": 0, "modulation": 0, "mode": "PLAY",
})
engine3.panic()
assert engine3.active_note_count == 0
print("  PASS: Panic releases all notes")

print()
print("=" * 60)
print("All Phase 5 tests passed.")
print("=" * 60)
