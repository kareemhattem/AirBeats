# =============================================================================
# AirBeats — Software Synthesizer
# music/synthesizer.py
#
# Generates audio in real time from musical events using numpy + sounddevice.
# No external MIDI libraries or FluidSynth required.
#
# ─────────────────────────────────────────────────────────────────────────────
# How it works
# ─────────────────────────────────────────────────────────────────────────────
#
# Each active note is a "Voice" — an object that produces audio samples for one
# note. A voice has:
#   - A MIDI note number (determines frequency)
#   - A velocity (determines amplitude)
#   - An ADSR envelope (Attack/Decay/Sustain/Release shape)
#   - An oscillator type (sine, saw, square, FM)
#   - Optional vibrato (from modulation depth)
#
# The Synthesizer holds a list of active voices. On each audio callback:
#   1. For each voice, generate N samples
#   2. Apply the ADSR envelope
#   3. Apply pitch bend (frequency ratio) and vibrato (LFO)
#   4. Mix all voices together
#   5. Apply a soft limiter to prevent clipping
#
# The audio output runs on its own thread via sounddevice's callback mechanism.
# The main thread calls synthesizer.note_on(), note_off(), set_pitch_bend(),
# and set_modulation() which are thread-safe (protected by a lock).
#
# ─────────────────────────────────────────────────────────────────────────────
# ADSR Envelope
# ─────────────────────────────────────────────────────────────────────────────
#
# The envelope shapes the amplitude of a note over its lifetime:
#
#   Attack  : 0 → peak      (rise time, e.g. 10 ms for plucked, 100 ms for pad)
#   Decay   : peak → sustain (fall time)
#   Sustain : held at sustain level while note is on
#   Release : sustain → 0    (fall time after note-off)
#
#   Amplitude
#       │  /\
#       │ /  \_______
#       │/            \
#       └──────────────\___ 0
#        A   D   S   R
#
# This is the standard subtractive synthesis envelope used in every synthesizer.
# =============================================================================

import numpy as np
import sounddevice as sd
import threading
import math
import time

import config


# =============================================================================
# Oscillator functions — generate one cycle of a waveform at a given frequency
# =============================================================================

def _sine_wave(phase: np.ndarray) -> np.ndarray:
    """Pure sine wave. Smoothest, least harmonics. Good for flutes and pads."""
    return np.sin(phase)


def _saw_wave(phase: np.ndarray) -> np.ndarray:
    """Sawtooth wave. Rich in harmonics. Good for brass and strings."""
    # numpy sawtooth: 2 * (phase / (2*pi) - floor(0.5 + phase / (2*pi)))
    return 2.0 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))


def _square_wave(phase: np.ndarray) -> np.ndarray:
    """Square wave. Hollow, reedy. Good for retro sounds and bass."""
    return np.sign(np.sin(phase))


def _fm_wave(phase: np.ndarray, carrier_freq: float, mod_index: float = 2.0,
             mod_ratio: float = 2.0, fs: float = 44100) -> np.ndarray:
    """
    FM synthesis: modulate the carrier frequency with a modulator.
    Produces rich, bell-like tones.

    y = sin(2*pi*carrier*t + mod_index * sin(2*pi*modulator*t))
    """
    t = phase / (2 * np.pi * carrier_freq)
    modulator = mod_index * np.sin(2 * np.pi * carrier_freq * mod_ratio * t)
    return np.sin(phase + modulator)


# Instrument presets — each defines oscillator type + ADSR + parameters
INSTRUMENT_PRESETS = {
    "piano": {
        "osc": "sine",
        "attack": 0.005,   # 5 ms — fast strike
        "decay": 0.3,      # 300 ms — natural decay
        "sustain": 0.3,    # 30% of peak while held
        "release": 0.4,    # 400 ms — tail
        "vibrato_depth_cents": 0,   # no vibrato
        "label": "Piano",
    },
    "pad": {
        "osc": "sine",
        "attack": 0.2,     # 200 ms — slow swell
        "decay": 0.1,
        "sustain": 0.8,    # high sustain — long notes
        "release": 0.5,    # 500 ms
        "vibrato_depth_cents": 10,
        "label": "Pad",
    },
    "pluck": {
        "osc": "saw",
        "attack": 0.002,   # 2 ms — instant strike
        "decay": 0.15,     # 150 ms — quick decay
        "sustain": 0.15,   # low sustain — notes fade
        "release": 0.1,
        "vibrato_depth_cents": 0,
        "label": "Pluck",
    },
    "bass": {
        "osc": "saw",
        "attack": 0.01,
        "decay": 0.05,
        "sustain": 0.7,
        "release": 0.15,
        "vibrato_depth_cents": 0,
        "label": "Bass",
    },
    "bell": {
        "osc": "fm",
        "attack": 0.001,
        "decay": 0.5,
        "sustain": 0.1,
        "release": 0.8,
        "vibrato_depth_cents": 0,
        "label": "Bell",
    },
}

INSTRUMENT_NAMES = list(INSTRUMENT_PRESETS.keys())


# =============================================================================
# Voice — one active note
# =============================================================================

class Voice:
    """
    A single playing note with its own oscillator and envelope state.

    Created on note_on(), destroyed after the release phase completes.
    """

    # Counter for unique voice IDs (useful for debugging)
    _next_id = 0

    def __init__(self, midi_note: int, velocity: int, instrument: dict,
                 fs: float):
        Voice._next_id += 1
        self.id          = Voice._next_id
        self.midi_note   = midi_note
        self.velocity    = velocity
        self.instrument  = instrument
        self.fs          = fs

        # Frequency from MIDI note number
        # f = 440 * 2^((note - 69) / 12)
        self.base_freq   = 440.0 * 2.0 ** ((midi_note - 69) / 12.0)

        # Amplitude from velocity (1-127 → 0.0-1.0)
        self.amplitude   = (velocity / 127.0) ** 1.5  # exponential curve

        # Envelope state
        self._env_phase  = "attack"   # attack → decay → sustain → release → done
        self._env_time   = 0.0
        self._env_value  = 0.0

        # Oscillator phase (accumulates over time)
        self._phase      = 0.0

        # Vibrato LFO phase
        self._vib_phase   = 0.0

        # Pitch bend ratio (1.0 = no bend)
        self._pitch_ratio = 1.0

        # Modulation depth (0-1)
        self._mod_depth    = 0.0

        # Note off flag
        self._released     = False

    def note_off(self):
        """Start the release phase."""
        if self._env_phase != "release" and self._env_phase != "done":
            self._env_phase = "release"
            self._env_time  = 0.0

    @property
    def is_done(self) -> bool:
        return self._env_phase == "done"

    def update_controls(self, pitch_bend: int, modulation: int):
        """Update pitch bend and modulation (called from the main thread)."""
        # MIDI pitch bend: -8192 to +8191, maps to ±2 semitones by default
        semitones = (pitch_bend / 8192.0) * config.PITCH_BEND_SEMITONES
        self._pitch_ratio = 2.0 ** (semitones / 12.0)
        self._mod_depth = modulation / 127.0

    def render(self, num_samples: int) -> np.ndarray:
        """
        Generate `num_samples` audio samples for this voice.

        Returns a float32 array in [-1, 1] range.
        """
        out = np.zeros(num_samples, dtype=np.float32)
        inst = self.instrument

        for i in range(num_samples):
            dt = 1.0 / self.fs

            # --- Envelope ---
            self._env_time += dt
            atk = inst["attack"]
            dec = inst["decay"]
            sus = inst["sustain"]
            rel = inst["release"]

            if self._env_phase == "attack":
                if atk <= 0:
                    self._env_value = 1.0
                    self._env_phase = "decay"
                    self._env_time = 0.0
                else:
                    self._env_value = self._env_time / atk
                    if self._env_time >= atk:
                        self._env_value = 1.0
                        self._env_phase = "decay"
                        self._env_time = 0.0

            elif self._env_phase == "decay":
                if dec <= 0:
                    self._env_value = sus
                    self._env_phase = "sustain"
                else:
                    self._env_value = 1.0 - (1.0 - sus) * (self._env_time / dec)
                    if self._env_time >= dec:
                        self._env_value = sus
                        self._env_phase = "sustain"

            elif self._env_phase == "sustain":
                self._env_value = sus

            elif self._env_phase == "release":
                if rel <= 0:
                    self._env_value = 0.0
                    self._env_phase = "done"
                else:
                    # Start release from current value, not sustain
                    self._env_value = max(0.0, self._env_value - (self._env_value * dt / rel))
                    if self._env_time >= rel or self._env_value < 0.001:
                        self._env_value = 0.0
                        self._env_phase = "done"

            # --- Vibrato (LFO) ---
            vib_depth_cents = inst.get("vibrato_depth_cents", 0)
            if vib_depth_cents > 0 and self._mod_depth > 0:
                vib_hz = 5.0  # 5 Hz vibrato rate
                self._vib_phase += 2 * np.pi * vib_hz * dt
                vib_semitones = (vib_depth_cents / 100.0) * self._mod_depth
                vib_ratio = 2.0 ** (vib_semitones * np.sin(self._vib_phase) / 12.0)
            else:
                vib_ratio = 1.0

            # --- Oscillator ---
            freq = self.base_freq * self._pitch_ratio * vib_ratio
            self._phase += 2 * np.pi * freq * dt
            if self._phase > 2 * np.pi:
                self._phase -= 2 * np.pi * np.floor(self._phase / (2 * np.pi))

            osc_type = inst["osc"]
            if osc_type == "sine":
                sample = np.sin(self._phase)
            elif osc_type == "saw":
                sample = 2.0 * (self._phase / (2 * np.pi) -
                                np.floor(0.5 + self._phase / (2 * np.pi)))
            elif osc_type == "square":
                sample = np.sign(np.sin(self._phase))
            elif osc_type == "fm":
                # Simple FM: modulate phase
                mod_sig = 2.0 * np.sin(self._phase * 2.0)  # ratio 2:1
                sample = np.sin(self._phase + mod_sig)
            else:
                sample = np.sin(self._phase)

            # Apply envelope and amplitude
            out[i] = sample * self._env_value * self.amplitude

        return out


# =============================================================================
# Synthesizer — manages voices and audio output
# =============================================================================

class Synthesizer:
    """
    Polyphonic software synthesizer with real-time audio output.

    Usage:
        synth = Synthesizer()
        synth.start()              # Opens audio output stream
        synth.set_instrument("piano")

        # Called from the acquisition thread:
        synth.note_on(60, velocity=100)   # MIDI note 60 = C4
        synth.note_off(60)
        synth.set_pitch_bend(2000)        # -8192 to +8191
        synth.set_modulation(64)          # 0-127

        synth.stop()               # Closes audio output
    """

    def __init__(self, sample_rate: int = 44100, blocksize: int = 256,
                 instrument: str = None):
        self._fs          = sample_rate
        self._blocksize   = blocksize
        self._instrument_name = instrument or config.DEFAULT_INSTRUMENT
        self._instrument  = INSTRUMENT_PRESETS[self._instrument_name].copy()

        self._voices      = []          # list of active Voice objects
        self._lock        = threading.Lock()

        self._stream       = None
        self._running      = False

        # Global controls (applied to all voices)
        self._pitch_bend   = 0
        self._modulation   = 0
        self._volume       = config.DEFAULT_VOLUME  # master volume 0-1

        # Diagnostics
        self._notes_played   = 0
        self._notes_active   = 0
        self._active_notes_set = set()   # MIDI note numbers currently on

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self):
        """Open the audio output stream and start playback."""
        try:
            self._stream = sd.OutputStream(
                samplerate=self._fs,
                blocksize=self._blocksize,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            print(f"[Synth] Audio started: {self._fs} Hz, block={self._blocksize}, "
                  f"instrument={self._instrument_name}")
        except Exception as e:
            print(f"[Synth] ERROR: Could not open audio output: {e}")
            self._running = False

    def stop(self):
        """Stop audio output and release resources."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._voices.clear()
            self._active_notes_set.clear()
        print("[Synth] Audio stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # -------------------------------------------------------------------------
    # Note control
    # -------------------------------------------------------------------------

    def note_on(self, midi_note: int, velocity: int):
        """
        Trigger a note. If the same note is already playing, retrigger it
        (release the old voice and create a new one).
        """
        with self._lock:
            # If this note is already active, release the old voice
            if midi_note in self._active_notes_set:
                for v in self._voices:
                    if v.midi_note == midi_note and not v._released:
                        v.note_off()
                        break

            # Create new voice
            voice = Voice(midi_note, velocity, self._instrument, self._fs)
            voice.update_controls(self._pitch_bend, self._modulation)
            self._voices.append(voice)
            self._active_notes_set.add(midi_note)
            self._notes_played += 1

    def note_off(self, midi_note: int):
        """Release a note. The voice enters its release phase."""
        with self._lock:
            if midi_note in self._active_notes_set:
                self._active_notes_set.discard(midi_note)
            for v in self._voices:
                if v.midi_note == midi_note and not v._released:
                    v.note_off()
                    break

    def all_notes_off(self):
        """Release all active notes immediately."""
        with self._lock:
            self._active_notes_set.clear()
            for v in self._voices:
                v.note_off()

    # -------------------------------------------------------------------------
    # Continuous controls
    # -------------------------------------------------------------------------

    def set_pitch_bend(self, bend: int):
        """Set pitch bend for all voices. Range: -8192 to +8191, 0 = none."""
        self._pitch_bend = max(-8192, min(8191, int(bend)))
        with self._lock:
            for v in self._voices:
                v.update_controls(self._pitch_bend, self._modulation)

    def set_modulation(self, mod: int):
        """Set modulation depth for all voices. Range: 0-127."""
        self._modulation = max(0, min(127, int(mod)))
        with self._lock:
            for v in self._voices:
                v.update_controls(self._pitch_bend, self._modulation)

    def set_volume(self, volume: float):
        """Set master output volume. Range: 0.0 to 1.0."""
        self._volume = max(0.0, min(1.0, volume))

    # -------------------------------------------------------------------------
    # Instrument selection
    # -------------------------------------------------------------------------

    def set_instrument(self, name: str):
        """Switch to a different instrument preset."""
        if name in INSTRUMENT_PRESETS:
            self._instrument_name = name
            self._instrument = INSTRUMENT_PRESETS[name].copy()
            print(f"[Synth] Instrument: {INSTRUMENT_PRESETS[name]['label']}")

    def cycle_instrument(self) -> str:
        """Cycle to the next instrument. Returns the new instrument name."""
        names = INSTRUMENT_NAMES
        idx = names.index(self._instrument_name) if self._instrument_name in names else 0
        idx = (idx + 1) % len(names)
        self.set_instrument(names[idx])
        return names[idx]

    @property
    def instrument_name(self) -> str:
        return self._instrument_name

    @property
    def active_note_count(self) -> int:
        return len(self._active_notes_set)

    @property
    def notes_played(self) -> int:
        return self._notes_played

    # -------------------------------------------------------------------------
    # Audio callback — called by sounddevice on its own thread
    # -------------------------------------------------------------------------

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        time_info, status):
        """
        Fill the output buffer with mixed audio from all active voices.

        This runs on sounddevice's audio thread, NOT the main thread.
        The lock ensures we don't modify the voice list while rendering.
        """
        if status.output_underflow:
            # Minor dropout — not critical, just continue
            pass

        # Start with silence
        outdata.fill(0)

        with self._lock:
            if not self._voices:
                return

            # Render and mix all voices
            voices_to_remove = []
            for v in self._voices:
                samples = v.render(frames)
                outdata[:, 0] += samples

                if v.is_done:
                    voices_to_remove.append(v)

            # Clean up finished voices
            for v in voices_to_remove:
                self._voices.remove(v)

        # Apply master volume and soft limiting
        outdata[:, 0] *= self._volume

        # Soft limiter: tanh saturates gracefully above ~1.0
        # This prevents harsh clipping distortion when many notes play at once
        if len(self._voices) > 2:
            outdata[:, 0] = np.tanh(outdata[:, 0] * 0.7)
