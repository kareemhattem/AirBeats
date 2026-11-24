# End-to-end lifecycle proof: bend -> audible note -> straighten -> release.
# Uses the REAL Synthesizer audio callback (not a mock), with a simulated
# DAC output buffer so no speaker is required for automated verification.
import sys, os, math, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import config
from processing.calibration import Calibration
from processing.filters import SignalProcessor, EMAFilter
from sensor_fusion.fusion import SensorFusion
from gesture_recognition.gestures import GestureRecognizer
from music.mapping import MusicalMapping
from music.midi_engine import MidiEngine
from music.synthesizer import Synthesizer

# ── Real pipeline objects ─────────────────────────────────────────────────────
cal = Calibration()
proc = SignalProcessor(cal)
proc.set_filter(EMAFilter, alpha=0.3)
fusion = SensorFusion(sample_rate_hz=50, alpha=0.98)
recog = GestureRecognizer()
mapping = MusicalMapping()

synth = Synthesizer()
engine = MidiEngine(synth)

# Start the real audio stream (callback mode) so voices render through the
# actual mixing path. Audio goes to the default output device — a short
# 261 Hz note WILL be audible during this test.
synth.start()
assert synth.is_running, "Synthesizer failed to start"

# ── Simulated DAC capture: intercept the audio callback output ───────────────
# We call the private callback directly with our own buffer, which exercises
# the exact mixing code the sound device runs.
captured = []

class _NullStatus:
    output_underflow = False

def run_dac_blocks(n_blocks):
    """Run the real audio callback n times, capturing the mixed output."""
    for _ in range(n_blocks):
        buf = np.zeros((synth._blocksize, 1), dtype=np.float32)
        synth._audio_callback(buf, synth._blocksize, None, _NullStatus())
        captured.append(np.max(np.abs(buf)))

# ── Feed the pipeline ─────────────────────────────────────────────────────────
def feed(flex_raw, pressure=0, label=""):
    raw = {
        "timestamp_ms": 0, "pressure": pressure,
        "flex": list(flex_raw),
        "accelerometer": [-0.038, -0.997, 0.065],  # resting hand
        "gyroscope": [0.0, 0.0, 0.0],
    }
    processed = proc.process(raw)
    processed.update(fusion.update(processed))
    g = recog.update(processed)
    processed.update({"gesture_id": g["id"], "gesture_name": g["name"],
                      "gesture_confidence": 0})
    ms = mapping.update(processed)
    engine.update(ms)
    if label:
        fp = processed["flex_pct"]
        print(f"  {label}: flex1={fp[0]:.2f}  roll={processed['roll']:+.1f}  "
              f"octave={ms['octave_shift']:+d}  "
              f"on={[n['midi_note'] for n in ms['active_notes']]}  "
              f"off={[n['midi_note'] for n in ms['released_notes']]}  "
              f"held={mapping._finger_active[0]}")
    return ms

# Warm up: 60 packets at rest (reference frame + filter settling)
for _ in range(60):
    feed([310, 300, 320, 305])

print("=" * 72)
print("NOTE LIFECYCLE TEST — real Synthesizer audio path")
print("=" * 72)

# ── Phase 1: silence baseline ─────────────────────────────────────────────────
run_dac_blocks(10)
baseline = max(captured)
print(f"\n  Baseline (no notes) DAC peak: {baseline:.6f}")

# ── Phase 2: bend finger 0 ───────────────────────────────────────────────────
# Ramp the ADC from 310 to 380 (crosses the 70% threshold at ~357)
captured.clear()
for i in range(1, 9):
    feed([310 + i * 9, 300, 320, 305])
    feed([310 + i * 9, 300, 320, 305])  # 2 feeds per step for hysteresis safety
feed([380, 300, 320, 305], label="BENT   ")
assert mapping._finger_active[0], "Finger 0 should be held (note on)"

# Let the note sound: run 0.3 s of audio blocks
run_dac_blocks(int(44100 / synth._blocksize * 0.3))
during_peak = max(captured)
print(f"  While bent: DAC peak = {during_peak:.5f}  "
      f"({'AUDIBLE' if during_peak > 0.01 else 'TOO QUIET'})")
assert during_peak > 0.01, f"Note not audible: peak {during_peak}"

# Verify the note number in the synth's active set
assert synth.active_note_count >= 1, "Synth should hold an active note"
active_note = next(iter(synth._active_notes_set))
print(f"  Active MIDI note in synth: {active_note} "
      f"({mapping._note_name(active_note)} = {440*2**((active_note-69)/12):.1f} Hz)")
assert active_note == 60, f"Expected C4 (60) at resting roll, got {active_note}"

# ── Phase 3: hold and verify sustain ─────────────────────────────────────────
# Piano envelope: peak amplitude 0.0625 (vel 20) decays to sustain = 0.3 of
# that, times master volume 0.5 → sustained level ≈ 0.0625*0.3*0.5 = 0.0094.
# The note remains audible; we assert it is still clearly above the silence
# floor (baseline) and does not decay to nothing.
captured.clear()
for _ in range(25):       # 0.5 s of packets, finger stays bent
    feed([380, 300, 320, 305])
run_dac_blocks(int(44100 / synth._blocksize * 0.5))
hold_peak = max(captured)
print(f"  Sustained hold: DAC peak = {hold_peak:.5f} (piano sustain envelope)")
assert hold_peak > 0.003, f"Note decayed to silence: peak {hold_peak}"

# ── Phase 4: straighten finger 0 ─────────────────────────────────────────────
captured.clear()
for i in range(1, 9):
    feed([380 - i * 9, 300, 320, 305])
    feed([380 - i * 9, 300, 320, 305])
feed([310, 300, 320, 305], label="RELEASE")
assert not mapping._finger_active[0], "Finger 0 should be released (note off)"

# Run 1.5 s of audio — enough for the 400 ms exponential release tail
# (tau = release = 0.4 s; 3.75 tau) to fall below the silence floor.
n_blocks = int(44100 / synth._blocksize * 1.5)
run_dac_blocks(n_blocks)
after_peak_any = max(captured)               # includes early release blocks
after_peak_end = max(captured[-40:])         # last ~230 ms of audio
print(f"  Release peak (early tail): {after_peak_any:.6f}")
print(f"  Release peak (final 0.25s): {after_peak_end:.6f} "
      f"({'SILENT' if after_peak_end < 0.002 else 'STILL SOUNDING'})")
assert after_peak_end < 0.002, f"Note did not release: final peak {after_peak_end}"

# ── Phase 5: no stuck notes ───────────────────────────────────────────────────
assert synth.active_note_count == 0, "No notes should remain active"
assert engine.active_note_count == 0, "MIDI engine should be empty"

synth.stop()
print()
print("=" * 72)
print("PASS: bend -> audible C4 (261.6 Hz) -> sustain -> straighten -> silence")
print("=" * 72)
